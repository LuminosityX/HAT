import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import glob
import json
import tqdm
import functools
import numpy as np

from torch.utils.data.distributed import DistributedSampler
from einops import rearrange

from .dist_utils import all_gather

from evaluation import i2t_SCAN, t2i_SCAN
import sys


def compute_mlm(pl_module, batch):
    infer = pl_module.infer(batch, mask_text=True, mask_image=False)
    mlm_logits = pl_module.mlm_score(infer["text_feats"])
    mlm_labels = infer["text_labels"]

    mlm_loss = F.cross_entropy(
        mlm_logits.view(-1, pl_module.hparams.config["vocab_size"]),
        mlm_labels.view(-1),
        ignore_index=-100,
    )

    ret = {
        "mlm_loss": mlm_loss,
        "mlm_logits": mlm_logits,
        "mlm_labels": mlm_labels,
        "mlm_ids": infer["text_ids"],
    }

    phase = "train" if pl_module.training else "val"
    loss = getattr(pl_module, f"{phase}_mlm_loss")(ret["mlm_loss"])
    acc = getattr(pl_module, f"{phase}_mlm_accuracy")(
        ret["mlm_logits"], ret["mlm_labels"]
    )
    pl_module.log(f"mlm/{phase}/loss", loss)
    pl_module.log(f"mlm/{phase}/accuracy", acc)

    return ret

def compute_itm(pl_module, batch):
    pos_len = len(batch["text"]) // 2
    neg_len = len(batch["text"]) - pos_len
    itm_labels = torch.cat([torch.ones(pos_len), torch.zeros(neg_len)]).to(
        pl_module.device
    )
    itm_labels = itm_labels[torch.randperm(itm_labels.size(0))]

    itm_images = [
        torch.stack(
            [
                ti if itm_labels[i] == 1 else fi
                for i, (ti, fi) in enumerate(zip(bti, bfi))
            ]
        )
        for bti, bfi in zip(batch["image"], batch["false_image_0"])
    ]

    batch = {k: v for k, v in batch.items()}
    batch["image"] = itm_images

    infer = pl_module.infer(batch, mask_text=False, mask_image=False)

    itm_logits = pl_module.itm_score(infer["cls_feats"])
    itm_loss = F.cross_entropy(itm_logits, itm_labels.long())

    ret = {
        "itm_loss": itm_loss,
        "itm_logits": itm_logits,
        "itm_labels": itm_labels,
    }

    phase = "train" if pl_module.training else "val"
    loss = getattr(pl_module, f"{phase}_itm_loss")(ret["itm_loss"])
    acc = getattr(pl_module, f"{phase}_itm_accuracy")(
        ret["itm_logits"], ret["itm_labels"]
    )
    pl_module.log(f"itm/{phase}/loss", loss)
    pl_module.log(f"itm/{phase}/accuracy", acc)

    return ret

def compute_snli(pl_module, batch):
    infer = pl_module.infer(
        batch, mask_text=False, mask_image=False, 
    )
    snli_logits = pl_module.snli_classifier(infer["cls_feats"])

    snli_labels = batch["labels"]
    snli_labels = torch.tensor(snli_labels).to(pl_module.device).long()
    snli_loss = F.cross_entropy(snli_logits, snli_labels.view(-1))

    ret = {
        "snli_loss": snli_loss,
        "snli_logits": snli_logits,
        "snli_labels": snli_labels,
    }

    phase = "train" if pl_module.training else "val"

    if phase == "train":
        loss = getattr(pl_module, f"{phase}_snli_loss")(ret["snli_loss"])
        acc = getattr(pl_module, f"{phase}_snli_accuracy")(
            ret["snli_logits"], ret["snli_labels"]
        )
        pl_module.log(f"snli/{phase}/loss", loss)
        pl_module.log(f"snli/{phase}/accuracy", acc)
    else:
        dev_batches = [i for i, n in enumerate(batch["table_name"]) if "dev" in n]
        test_batches = [i for i, n in enumerate(batch["table_name"]) if "test" in n]

        if dev_batches:
            dev_loss = getattr(pl_module, f"dev_snli_loss")(
                F.cross_entropy(
                    ret["snli_logits"][dev_batches], ret["snli_labels"][dev_batches]
                )
            )
            dev_acc = getattr(pl_module, f"dev_snli_accuracy")(
                ret["snli_logits"][dev_batches], ret["snli_labels"][dev_batches]
            )
            pl_module.log(f"snli/dev/loss", dev_loss)
            pl_module.log(f"snli/dev/accuracy", dev_acc)
        if test_batches:
            test_loss = getattr(pl_module, f"test_snli_loss")(
                F.cross_entropy(
                    ret["snli_logits"][test_batches], ret["snli_labels"][test_batches]
                )
            )
            test_acc = getattr(pl_module, f"test_snli_accuracy")(
                ret["snli_logits"][test_batches], ret["snli_labels"][test_batches]
            )
            pl_module.log(f"snli/test/loss", test_loss)
            pl_module.log(f"snli/test/accuracy", test_acc)

    return ret

def compute_vqa(pl_module, batch):
    infer = pl_module.infer(batch, mask_text=False, mask_image=False)
    vqa_logits = pl_module.vqa_classifier(infer["cls_feats"])
    vqa_targets = torch.zeros(
        len(vqa_logits), pl_module.hparams.config["vqav2_label_size"]
    ).to(pl_module.device)

    vqa_labels = batch["vqa_labels"]
    vqa_scores = batch["vqa_scores"]

    for i, (_label, _score) in enumerate(zip(vqa_labels, vqa_scores)):
        for l, s in zip(_label, _score):
            vqa_targets[i, l] = s

    vqa_loss = (
        F.binary_cross_entropy_with_logits(vqa_logits, vqa_targets)
        * vqa_targets.shape[1]
    )  # https://github.com/jnhwkim/ban-vqa/blob/master/train.py#L19

    ret = {
        "vqa_loss": vqa_loss,
        "vqa_logits": vqa_logits,
        "vqa_targets": vqa_targets,
        "vqa_labels": vqa_labels,
        "vqa_scores": vqa_scores,
    }

    phase = "train" if pl_module.training else "val"
    loss = getattr(pl_module, f"{phase}_vqa_loss")(ret["vqa_loss"])
    score = getattr(pl_module, f"{phase}_vqa_score")(
        ret["vqa_logits"], ret["vqa_targets"]
    )
    pl_module.log(f"vqa/{phase}/loss", loss)
    pl_module.log(f"vqa/{phase}/score", score)

    return ret


def compute_nlvr2(pl_module, batch):
    infer1 = pl_module.infer(
        batch, mask_text=False, mask_image=False, image_token_type_idx=1
    )
    infer2 = pl_module.infer(
        batch, mask_text=False, mask_image=False, image_token_type_idx=2
    )

    cls_feats = torch.cat([infer1["cls_feats"], infer2["cls_feats"]], dim=-1)
    nlvr2_logits = pl_module.nlvr2_classifier(cls_feats)

    nlvr2_labels = batch["answers"]
    nlvr2_labels = torch.tensor(nlvr2_labels).to(pl_module.device).long()
    nlvr2_loss = F.cross_entropy(nlvr2_logits, nlvr2_labels.view(-1))

    ret = {
        "nlvr2_loss": nlvr2_loss,
        "nlvr2_logits": nlvr2_logits,
        "nlvr2_labels": nlvr2_labels,
    }

    phase = "train" if pl_module.training else "val"

    if phase == "train":
        loss = getattr(pl_module, f"{phase}_nlvr2_loss")(ret["nlvr2_loss"])
        acc = getattr(pl_module, f"{phase}_nlvr2_accuracy")(
            ret["nlvr2_logits"], ret["nlvr2_labels"]
        )
        pl_module.log(f"nlvr2/{phase}/loss", loss)
        pl_module.log(f"nlvr2/{phase}/accuracy", acc)
    else:
        dev_batches = [i for i, n in enumerate(batch["table_name"]) if "dev" in n]
        test_batches = [i for i, n in enumerate(batch["table_name"]) if "test" in n]

        if dev_batches:
            dev_loss = getattr(pl_module, f"dev_nlvr2_loss")(
                F.cross_entropy(
                    ret["nlvr2_logits"][dev_batches], ret["nlvr2_labels"][dev_batches]
                )
            )
            dev_acc = getattr(pl_module, f"dev_nlvr2_accuracy")(
                ret["nlvr2_logits"][dev_batches], ret["nlvr2_labels"][dev_batches]
            )
            pl_module.log(f"nlvr2/dev/loss", dev_loss)
            pl_module.log(f"nlvr2/dev/accuracy", dev_acc)
        if test_batches:
            test_loss = getattr(pl_module, f"test_nlvr2_loss")(
                F.cross_entropy(
                    ret["nlvr2_logits"][test_batches], ret["nlvr2_labels"][test_batches]
                )
            )
            test_acc = getattr(pl_module, f"test_nlvr2_accuracy")(
                ret["nlvr2_logits"][test_batches], ret["nlvr2_labels"][test_batches]
            )
            pl_module.log(f"nlvr2/test/loss", test_loss)
            pl_module.log(f"nlvr2/test/accuracy", test_acc)

    return ret


def compute_irtr(pl_module, batch):
    is_training_phase = pl_module.training

    _bs, _c, _h, _w = batch["image"][0].shape
    false_len = pl_module.hparams.config["draw_false_text"]
    text_ids = torch.stack(
        [batch[f"false_text_{i}_ids"] for i in range(false_len)], dim=1
    )
    text_masks = torch.stack(
        [batch[f"false_text_{i}_masks"] for i in range(false_len)], dim=1
    )
    text_labels = torch.stack(
        [batch[f"false_text_{i}_labels"] for i in range(false_len)], dim=1
    )

    text_ids = torch.cat([batch["text_ids"].unsqueeze(1), text_ids], dim=1)
    text_masks = torch.cat([batch["text_masks"].unsqueeze(1), text_masks], dim=1)
    text_labels = torch.cat([batch["text_labels"].unsqueeze(1), text_labels], dim=1)
    images = batch["image"][0].unsqueeze(1).expand(_bs, false_len + 1, _c, _h, _w)

    infer = pl_module.infer(
        {
            "image": [rearrange(images, "bs fs c h w -> (bs fs) c h w")],
            "text_ids": rearrange(text_ids, "bs fs tl -> (bs fs) tl"),
            "text_masks": rearrange(text_masks, "bs fs tl -> (bs fs) tl"),
            "text_labels": rearrange(text_labels, "bs fs tl -> (bs fs) tl"),
        }
    )
    score = pl_module.rank_output(infer["cls_feats"])[:, 0]
    score = rearrange(score, "(bs fs) -> bs fs", bs=_bs, fs=false_len + 1)
    answer = torch.zeros(_bs).to(score).long()
    irtr_loss = F.cross_entropy(score, answer)

    ret = {
        "irtr_loss": irtr_loss,
    }

    phase = "train" if pl_module.training else "val"
    irtr_loss = getattr(pl_module, f"{phase}_irtr_loss")(ret["irtr_loss"])

    pl_module.log(f"irtr/{phase}/irtr_loss", irtr_loss)

    return ret

def compute_contrastiveLoss(im, s, margin):
    # compute image-sentence score matrix
        scores = im.mm(s.t())
        diagonal = scores.diag().view(im.size(0), 1)
        d1 = diagonal.expand_as(scores)
        d2 = diagonal.t().expand_as(scores)

        # compare every diagonal score to scores in its column
        # caption retrieval
        cost_s = (margin + scores - d1).clamp(min=0)
        # compare every diagonal score to scores in its row
        # image retrieval
        cost_im = (margin + scores - d2).clamp(min=0)

        # clear diagonals
        mask = torch.eye(scores.size(0)) > .5
        I = mask
        if torch.cuda.is_available():
            I = I.cuda()
        cost_s = cost_s.masked_fill_(I, 0)
        cost_im = cost_im.masked_fill_(I, 0)

        # keep the maximum violating negative for each query
        
        cost_s = cost_s.max(1)[0]
        cost_im = cost_im.max(0)[0]

        return cost_s.sum() + cost_im.sum()

def l2norm(X, dim, eps=1e-8):
    """L2-normalize columns of X
    """
    norm = torch.pow(X, 2).sum(dim=dim, keepdim=True).sqrt() + eps
    X = torch.div(X, norm)
    return X

def focal_equal(attn, batch_size, queryL, sourceL):
    """
    consider the confidence g(x) for each fragment as equal
    sigma_{j} (xi - xj) = sigma_{j} xi - sigma_{j} xj
    attn: (batch, queryL, sourceL)
    """
    funcF = attn * sourceL - torch.sum(attn, dim=-1, keepdim=True)
    fattn = torch.where(funcF > 0, torch.ones_like(attn),
                        torch.zeros_like(attn))
    return fattn

def func_attention(query, context, smooth, eps=1e-8):
    """
    query: (n_context, queryL, d)
    context: (n_context, sourceL, d)
    """
    batch_size_q, queryL = query.size(0), query.size(1)
    batch_size, sourceL = context.size(0), context.size(1)


    # Get attention
    # --> (batch, d, queryL)
    queryT = torch.transpose(query, 1, 2)

    # (batch, sourceL, d)(batch, d, queryL)
    # --> (batch, sourceL, queryL)
    attn = torch.bmm(context, queryT)
    
    attn = nn.LeakyReLU(0.1)(attn)
    attn = l2norm(attn, 2)
    # --> (batch, queryL, sourceL)
    attn = torch.transpose(attn, 1, 2).contiguous()
    # --> (batch*queryL, sourceL)
    attn = attn.view(batch_size*queryL, sourceL)
    attn = nn.Softmax(1)(attn*smooth)
    # --> (batch, queryL, sourceL)
    attn = attn.view(batch_size, queryL, sourceL)
    # --> (batch, sourceL, queryL)
    attnT = torch.transpose(attn, 1, 2).contiguous()
    #print(attnT.shape)

    #pic = attnT[0][0].view(28, 28)
    #print(pic)
    #plt.matshow(pic.data.cpu().numpy(), cmap=plt.cm.Blues)
    #plt.savefig('3.jpg')
    #assert 1==0

    # --> (batch, d, sourceL)
    contextT = torch.transpose(context, 1, 2)
    # (batch x d x sourceL)(batch x sourceL x queryL)
    # --> (batch, d, queryL)
    weightedContext = torch.bmm(contextT, attnT)
    # --> (batch, queryL, d)
    weightedContext = torch.transpose(weightedContext, 1, 2)

    return weightedContext, attnT


'''#BFAN
def func_attention(query, context, smooth, eps=1e-8):
    """
    query: (n_context, queryL, d)
    context: (n_context, sourceL, d)
    """
    batch_size_q, queryL = query.size(0), query.size(1)
    batch_size, sourceL = context.size(0), context.size(1)


    # Get attention
    # --> (batch, d, queryL)
    queryT = torch.transpose(query, 1, 2)

    # (batch, sourceL, d)(batch, d, queryL)
    # --> (batch, sourceL, queryL)
    attn = torch.bmm(context, queryT)
    
    attn = nn.LeakyReLU(0.1)(attn)
    attn = l2norm(attn, 2)
    # --> (batch, queryL, sourceL)
    attn = torch.transpose(attn, 1, 2).contiguous()
    # --> (batch*queryL, sourceL)
    attn = attn.view(batch_size*queryL, sourceL)
    attn = nn.Softmax(1)(attn*smooth)
    # --> (batch, queryL, sourceL)
    attn = attn.view(batch_size, queryL, sourceL)

    #BFAN

    funcH = focal_equal(attn, batch_size, queryL, sourceL)

    tmp_attn = funcH * attn
    attn_sum = torch.sum(tmp_attn, dim=-1, keepdim=True)
    re_attn = tmp_attn / attn_sum



    # --> (batch, sourceL, queryL)
    re_attnT = torch.transpose(re_attn, 1, 2).contiguous()

    # --> (batch, d, sourceL)
    contextT = torch.transpose(context, 1, 2)
    # (batch x d x sourceL)(batch x sourceL x queryL)
    # --> (batch, d, queryL)
    weightedContext = torch.bmm(contextT, re_attnT)
    # --> (batch, queryL, d)
    weightedContext = torch.transpose(weightedContext, 1, 2)

    return weightedContext, re_attnT'''


def cosine_similarity(x1, x2, dim=1, eps=1e-8):
    """Returns cosine similarity between x1 and x2, computed along dim."""
    w12 = torch.sum(x1 * x2, dim)
    w1 = torch.norm(x1, 2, dim)
    w2 = torch.norm(x2, 2, dim)
    return (w12 / (w1 * w2).clamp(min=eps)).squeeze()

def xattn_score_i2t(images, captions, cap_lens, lambda_softmax):
    """
    Images: (batch_size, n_regions, d) matrix of images
    Captions: (batch_size, max_n_words, d) matrix of captions
    CapLens: (batch_size) array of caption lengths
    """
    similarities = []
    n_image = images.size(0)
    n_caption = captions.size(0)
    n_region = images.size(1)
    for i in range(n_caption):
        # Get the i-th text description
        n_word = cap_lens[i]
        cap_i = captions[i, :n_word, :].unsqueeze(0).contiguous()
        # (n_image, n_word, d)
        cap_i_expand = cap_i.repeat(n_image, 1, 1)
        """
            word(query): (n_image, n_word, d)
            image(context): (n_image, n_region, d)
            weiContext: (n_image, n_region, d)
            attn: (n_image, n_word, n_region)
        """
        weiContext, attn = func_attention(images, cap_i_expand, smooth=lambda_softmax)
        # (n_image, n_region)
        row_sim = cosine_similarity(images, weiContext, dim=2)
        
        row_sim = row_sim.mean(dim=1, keepdim=True)
        similarities.append(row_sim)

    # (n_image, n_caption)
    similarities = torch.cat(similarities, 1)
    return similarities

def xattn_score_t2i(images, captions, cap_lens, lambda_softmax):
    """
    Images: (n_image, n_regions, d) matrix of images
    Captions: (n_caption, max_n_word, d) matrix of captions
    CapLens: (n_caption) array of caption lengths
    """
    similarities = []
    n_image = images.size(0)
    n_caption = captions.size(0)
    for i in range(n_caption):
        # Get the i-th text description
        n_word = cap_lens[i]
        cap_i = captions[i, :n_word, :].unsqueeze(0).contiguous()
        # --> (n_image, n_word, d)
        cap_i_expand = cap_i.repeat(n_image, 1, 1)
        """
            word(query): (n_image, n_word, d)
            image(context): (n_image, n_regions, d)
            weiContext: (n_image, n_word, d)
            attn: (n_image, n_region, n_word)
        """
        weiContext, attn = func_attention(cap_i_expand, images, smooth=lambda_softmax)
        cap_i_expand = cap_i_expand.contiguous()
        weiContext = weiContext.contiguous()
        # (n_image, n_word)
        row_sim = cosine_similarity(cap_i_expand, weiContext, dim=2)
        
        row_sim = row_sim.mean(dim=1, keepdim=True)
        
        similarities.append(row_sim)

    # (n_image, n_caption)
    similarities = torch.cat(similarities, 1)
    
    return similarities

def compute_SCAN(im, s, s_l, add_data, margin, direction, lambda_softmax):

        (image_feats_1, text_feats_1, image_feats_2, text_feats_2) = add_data
        # compute image-sentence score matrix
        if direction == 't2i':
            scores = xattn_score_t2i(im, s, s_l, lambda_softmax)
            scores_1 = xattn_score_t2i(image_feats_1, text_feats_1, s_l, lambda_softmax)
            scores_2 = xattn_score_t2i(image_feats_2, text_feats_2, s_l, lambda_softmax)

            scores = (scores + scores_1 + scores_2)

        elif direction == 'i2t':
            scores = xattn_score_i2t(im, s, s_l, lambda_softmax)
            #scores_0 = xattn_score_i2t(image_feats_0, text_feats_0, s_l, lambda_softmax)
            scores_1 = xattn_score_i2t(image_feats_1, text_feats_1, s_l, lambda_softmax)
            scores_2 = xattn_score_i2t(image_feats_2, text_feats_2, s_l, lambda_softmax)

            scores = (scores + scores_1 + scores_2)
        else:
            raise ValueError("unknown first norm type")
        diagonal = scores.diag().view(im.size(0), 1)
        d1 = diagonal.expand_as(scores)
        d2 = diagonal.t().expand_as(scores)

        # compare every diagonal score to scores in its column
        # caption retrieval
        cost_s = (margin + scores - d1).clamp(min=0)
        # compare every diagonal score to scores in its row
        # image retrieval
        cost_im = (margin + scores - d2).clamp(min=0)

        # clear diagonals
        mask = torch.eye(scores.size(0)) > .5
        I = mask
        if torch.cuda.is_available():
            I = I.cuda()
        cost_s = cost_s.masked_fill_(I, 0)
        cost_im = cost_im.masked_fill_(I, 0)

        # keep the maximum violating negative for each query
        
        cost_s = cost_s.max(1)[0]
        cost_im = cost_im.max(0)[0]
        return cost_s.sum() + cost_im.sum()

def compute_irtr_my(pl_module, batch):
    images, input_ids, lengths, ids, input_mask, input_type_ids = batch
    infer = pl_module.infer(
        {
            "image": [images],
            "text_ids": input_ids,
            "text_masks": input_mask,
            "text_types": input_type_ids,
            'lengths': lengths,
        }
    )

    image_feats = infer['image_feats']  # (B 49 H) (when infer use befor pool)  after pool:  (B H)
    text_feats = infer['text_feats']    # (B 32 H)                                           (B H)
    text_lengths = infer['text_lengths']

    #

    #text_feats_0 = infer['text_feats_0']
    text_feats_1 = infer['text_feats_1']
    text_feats_2 = infer['text_feats_2']

    #image_feats_0 = infer['image_feats_0']
    image_feats_1 = infer['image_feats_1']
    image_feats_2 = infer['image_feats_2']

    '''v_cluster = infer['v_cluster']
    v_cluster_2 = infer['v_cluster_2']
    v_cluster_1 = infer['v_cluster_1']'''

    add_data = (image_feats_1, text_feats_1, image_feats_2, text_feats_2)
    
    #
    #loacl_loss = LCA(image_feats_1, text_feats_1, text_lengths)
    '''loss_lca = pl_module.lca_loss(v_cluster, text_feats, text_lengths)
    loss_lca_2 = pl_module.lca_loss_2(v_cluster_2, text_feats_2, text_lengths)
    loss_lca_1 = pl_module.lca_loss_1(v_cluster_1, text_feats_1, text_lengths)

    loss_lca = (loss_lca + loss_lca_2 + loss_lca_1) / 3'''


    #

    loss = compute_SCAN(image_feats, text_feats, text_lengths, add_data, pl_module.hparams.config["margin"], pl_module.hparams.config["direction"], pl_module.hparams.config["lambda_softmax"])

    ret = {
        "irtr_loss": loss,
        #"lca_loss": loss_lca,
    }

    return ret

def shard_xattn_i2t(images, captions, caplens, lambda_softmax, shard_size=128):
    """
    Computer pairwise i2t image-caption distance with locality sharding
    """
    n_im_shard = (len(images)-1)//shard_size + 1
    n_cap_shard = (len(captions)-1)//shard_size + 1
    
    d = np.zeros((len(images), len(captions)))
    for i in range(n_im_shard):
        im_start, im_end = shard_size*i, min(shard_size*(i+1), len(images))
        for j in range(n_cap_shard):
            sys.stdout.write('\r>> shard_xattn_i2t batch (%d,%d)' % (i,j))
            cap_start, cap_end = shard_size*j, min(shard_size*(j+1), len(captions))
            im = torch.from_numpy(images[im_start:im_end]).cuda()
            s = torch.from_numpy(captions[cap_start:cap_end]).cuda()
            l = caplens[cap_start:cap_end]
            sim = xattn_score_i2t(im, s, l, lambda_softmax)
            d[im_start:im_end, cap_start:cap_end] = sim.data.cpu().numpy()
    sys.stdout.write('\n')
    return d

def shard_xattn_t2i(images, captions, caplens, lambda_softmax, shard_size=128):
    """
    Computer pairwise t2i image-caption distance with locality sharding
    """
    n_im_shard = (len(images)-1)//shard_size + 1
    n_cap_shard = (len(captions)-1)//shard_size + 1
    
    d = np.zeros((len(images), len(captions)))
    for i in range(n_im_shard):
        im_start, im_end = shard_size*i, min(shard_size*(i+1), len(images))
        for j in range(n_cap_shard):
            sys.stdout.write('\r>> shard_xattn_t2i batch (%d,%d)' % (i,j))
            cap_start, cap_end = shard_size*j, min(shard_size*(j+1), len(captions))
            im = torch.from_numpy(images[im_start:im_end]).cuda()
            s = torch.from_numpy(captions[cap_start:cap_end]).cuda()
            l = caplens[cap_start:cap_end]
            sim = xattn_score_t2i(im, s, l, lambda_softmax)
            d[im_start:im_end, cap_start:cap_end] = sim.data.cpu().numpy()
    sys.stdout.write('\n')
    return d

@torch.no_grad()
def compute_irtr_val(pl_module):
    val_dataloader = pl_module.trainer.datamodule.val_dataloader()

    img_embs = None
    cap_embs = None

    #img_embs_0 = None
    #cap_embs_0 = None
    img_embs_1 = None
    cap_embs_1 = None
    img_embs_2 = None
    cap_embs_2 = None

    stc_lens = None
    for (images, input_ids, lengths, ids, input_mask, input_type_ids) in val_dataloader:
        # make sure val logger is used

        # compute the embeddings
        infer = pl_module.infer({"image": [images.to(pl_module.device)],
                                            "text_ids": input_ids.to(pl_module.device),
                                            "text_masks": input_mask.to(pl_module.device),
                                            "text_types": input_type_ids.to(pl_module.device),
                                            'lengths': lengths,
        })

        img_emb = infer['image_feats']
        cap_emb = infer['text_feats']

        #

        #cap_emb_0 = infer['text_feats_0']
        cap_emb_1 = infer['text_feats_1']
        cap_emb_2 = infer['text_feats_2']

        #img_emb_0 = infer['image_feats_0']
        img_emb_1 = infer['image_feats_1']
        img_emb_2 = infer['image_feats_2']

        #

        # initialize the numpy arrays given the size of the embeddings
        if img_embs is None:
            img_embs = np.zeros((len(val_dataloader.dataset), img_emb.size(1), img_emb.size(2)))
            cap_embs = np.zeros((len(val_dataloader.dataset), cap_emb.size(1), cap_emb.size(2)))

            #

            #img_embs_0 = np.zeros((len(val_dataloader.dataset), img_emb_0.size(1), img_emb_0.size(2)))
            #cap_embs_0 = np.zeros((len(val_dataloader.dataset), cap_emb_0.size(1), cap_emb_0.size(2)))
            img_embs_1 = np.zeros((len(val_dataloader.dataset), img_emb_1.size(1), img_emb_1.size(2)))
            cap_embs_1 = np.zeros((len(val_dataloader.dataset), cap_emb_1.size(1), cap_emb_1.size(2)))
            img_embs_2 = np.zeros((len(val_dataloader.dataset), img_emb_2.size(1), img_emb_2.size(2)))
            cap_embs_2 = np.zeros((len(val_dataloader.dataset), cap_emb_2.size(1), cap_emb_2.size(2)))

            #

            stc_lens = np.zeros(len(val_dataloader.dataset), dtype=np.int)

        # preserve the embeddings by copying from gpu and converting to numpy
        img_embs[ids] = img_emb.data.cpu().numpy().copy()
        cap_embs[ids] = cap_emb.data.cpu().numpy().copy()

        #
        #img_embs_0[ids] = img_emb_0.data.cpu().numpy().copy()
        #cap_embs_0[ids] = cap_emb_0.data.cpu().numpy().copy()
        img_embs_1[ids] = img_emb_1.data.cpu().numpy().copy()
        cap_embs_1[ids] = cap_emb_1.data.cpu().numpy().copy()
        img_embs_2[ids] = img_emb_2.data.cpu().numpy().copy()
        cap_embs_2[ids] = cap_emb_2.data.cpu().numpy().copy()
        #

        stc_lens[ids] = np.asarray(lengths, dtype=np.int)

    img_embs = np.array([img_embs[i] for i in range(0, len(img_embs), 5)])
    
    #
    #img_embs_0 = np.array([img_embs_0[i] for i in range(0, len(img_embs_0), 5)])
    img_embs_1 = np.array([img_embs_1[i] for i in range(0, len(img_embs_1), 5)])
    img_embs_2 = np.array([img_embs_2[i] for i in range(0, len(img_embs_2), 5)])

    #

    '''np.save('/data2/lihaoxuan/New-Swin-Ablation/case/img.npy', img_embs)
    np.save('/data2/lihaoxuan/New-Swin-Ablation/case/txt.npy', cap_embs)

    np.save('/data2/lihaoxuan/New-Swin-Ablation/case/img_1.npy', img_embs_1)
    np.save('/data2/lihaoxuan/New-Swin-Ablation/case/txt_1.npy', cap_embs_1)

    np.save('/data2/lihaoxuan/New-Swin-Ablation/case/img_2.npy', img_embs_2)
    np.save('/data2/lihaoxuan/New-Swin-Ablation/case/txt_2.npy', cap_embs_2)

    np.save('/data2/lihaoxuan/New-Swin-Ablation/case/len.npy', stc_lens)'''

    #assert 1==0

    if pl_module.hparams.config["direction"] == 'i2t':
        sims = shard_xattn_i2t(img_embs, cap_embs, stc_lens, pl_module.hparams.config["lambda_softmax"], shard_size=128)
        #sims_0 = shard_xattn_i2t(img_embs_0, cap_embs_0, stc_lens, pl_module.hparams.config["lambda_softmax"], shard_size=128)
        sims_1 = shard_xattn_i2t(img_embs_1, cap_embs_1, stc_lens, pl_module.hparams.config["lambda_softmax"], shard_size=128)
        sims_2 = shard_xattn_i2t(img_embs_2, cap_embs_2, stc_lens, pl_module.hparams.config["lambda_softmax"], shard_size=128)
    else:
        sims = shard_xattn_t2i(img_embs, cap_embs, stc_lens, pl_module.hparams.config["lambda_softmax"], shard_size=128)
        #sims_0 = shard_xattn_t2i(img_embs_0, cap_embs_0, stc_lens, pl_module.hparams.config["lambda_softmax"], shard_size=128)
        sims_1 = shard_xattn_t2i(img_embs_1, cap_embs_1, stc_lens, pl_module.hparams.config["lambda_softmax"], shard_size=128)
        sims_2 = shard_xattn_t2i(img_embs_2, cap_embs_2, stc_lens, pl_module.hparams.config["lambda_softmax"], shard_size=128)
       

    

    sims_all = sims  + sims_1 + sims_2

    (r1, r5, r10, r20, r50, r70, r100, medr, meanr) = i2t_SCAN(sims)
    (r1i, r5i, r10i, r20i, r50i, r70i, r100i, medri, meanri) = t2i_SCAN(sims)

    print("Image to text: %.1f, %.1f, %.1f, %.1f, %.1f, %.1f, %.1f" % (r1, r5, r10, r20, r50, r70, r100))
        
    print("Text to image: %.1f, %.1f, %.1f, %.1f, %.1f, %.1f, %.1f" % (r1i, r5i, r10i, r20i, r50i, r70i, r100i))

    (r1, r5, r10, r20, r50, r70, r100, medr, meanr) = i2t_SCAN(sims_all)
    (r1i, r5i, r10i, r20i, r50i, r70i, r100i, medri, meanri) = t2i_SCAN(sims_all)

    #np.save('/data2/lihaoxuan/New-Swin-Ablation/test/flickr/runs/sims/i2t.npy', sims_all)
    
    pl_module.log('best_irtr', (r1+r1i))
        

    return (r1, r5, r10, r20, r50, r70, r100, r1i, r5i, r10i, r20i, r50i, r70i, r100i)

@torch.no_grad()
def compute_irtr_test(pl_module):
    val_dataloader = pl_module.trainer.datamodule.test_dataloader()

    img_embs = None
    cap_embs = None

    #img_embs_0 = None
    #cap_embs_0 = None
    img_embs_1 = None
    cap_embs_1 = None
    img_embs_2 = None
    cap_embs_2 = None

    stc_lens = None
    for (images, input_ids, lengths, ids, input_mask, input_type_ids) in val_dataloader:
        # make sure val logger is used

        # compute the embeddings
        infer = pl_module.infer({"image": [images.to(pl_module.device)],
                                            "text_ids": input_ids.to(pl_module.device),
                                            "text_masks": input_mask.to(pl_module.device),
                                            "text_types": input_type_ids.to(pl_module.device),
                                            'lengths': lengths,
        })

        img_emb = infer['image_feats']
        cap_emb = infer['text_feats']

        #

        #cap_emb_0 = infer['text_feats_0']
        cap_emb_1 = infer['text_feats_1']
        cap_emb_2 = infer['text_feats_2']

        #img_emb_0 = infer['image_feats_0']
        img_emb_1 = infer['image_feats_1']
        img_emb_2 = infer['image_feats_2']

        #

        # initialize the numpy arrays given the size of the embeddings
        if img_embs is None:
            img_embs = np.zeros((len(val_dataloader.dataset), img_emb.size(1), img_emb.size(2)))
            cap_embs = np.zeros((len(val_dataloader.dataset), cap_emb.size(1), cap_emb.size(2)))

            #

            #img_embs_0 = np.zeros((len(val_dataloader.dataset), img_emb_0.size(1), img_emb_0.size(2)))
            #cap_embs_0 = np.zeros((len(val_dataloader.dataset), cap_emb_0.size(1), cap_emb_0.size(2)))
            img_embs_1 = np.zeros((len(val_dataloader.dataset), img_emb_1.size(1), img_emb_1.size(2)))
            cap_embs_1 = np.zeros((len(val_dataloader.dataset), cap_emb_1.size(1), cap_emb_1.size(2)))
            img_embs_2 = np.zeros((len(val_dataloader.dataset), img_emb_2.size(1), img_emb_2.size(2)))
            cap_embs_2 = np.zeros((len(val_dataloader.dataset), cap_emb_2.size(1), cap_emb_2.size(2)))

            #

            stc_lens = np.zeros(len(val_dataloader.dataset), dtype=np.int)

        # preserve the embeddings by copying from gpu and converting to numpy
        img_embs[ids] = img_emb.data.cpu().numpy().copy()
        cap_embs[ids] = cap_emb.data.cpu().numpy().copy()

        #
        #img_embs_0[ids] = img_emb_0.data.cpu().numpy().copy()
        #cap_embs_0[ids] = cap_emb_0.data.cpu().numpy().copy()
        img_embs_1[ids] = img_emb_1.data.cpu().numpy().copy()
        cap_embs_1[ids] = cap_emb_1.data.cpu().numpy().copy()
        img_embs_2[ids] = img_emb_2.data.cpu().numpy().copy()
        cap_embs_2[ids] = cap_emb_2.data.cpu().numpy().copy()
        #

        stc_lens[ids] = np.asarray(lengths, dtype=np.int)

    img_embs = np.array([img_embs[i] for i in range(0, len(img_embs), 5)])
    
    #
    #img_embs_0 = np.array([img_embs_0[i] for i in range(0, len(img_embs_0), 5)])
    img_embs_1 = np.array([img_embs_1[i] for i in range(0, len(img_embs_1), 5)])
    img_embs_2 = np.array([img_embs_2[i] for i in range(0, len(img_embs_2), 5)])

    #

    '''np.save('/data2/lihaoxuan/New-Swin-Ablation/case/img.npy', img_embs)
    np.save('/data2/lihaoxuan/New-Swin-Ablation/case/txt.npy', cap_embs)

    np.save('/data2/lihaoxuan/New-Swin-Ablation/case/img_1.npy', img_embs_1)
    np.save('/data2/lihaoxuan/New-Swin-Ablation/case/txt_1.npy', cap_embs_1)

    np.save('/data2/lihaoxuan/New-Swin-Ablation/case/img_2.npy', img_embs_2)
    np.save('/data2/lihaoxuan/New-Swin-Ablation/case/txt_2.npy', cap_embs_2)

    np.save('/data2/lihaoxuan/New-Swin-Ablation/case/len.npy', stc_lens)'''

    #assert 1==0

    if pl_module.hparams.config["direction"] == 'i2t':
        sims = shard_xattn_i2t(img_embs, cap_embs, stc_lens, pl_module.hparams.config["lambda_softmax"], shard_size=128)
        #sims_0 = shard_xattn_i2t(img_embs_0, cap_embs_0, stc_lens, pl_module.hparams.config["lambda_softmax"], shard_size=128)
        sims_1 = shard_xattn_i2t(img_embs_1, cap_embs_1, stc_lens, pl_module.hparams.config["lambda_softmax"], shard_size=128)
        sims_2 = shard_xattn_i2t(img_embs_2, cap_embs_2, stc_lens, pl_module.hparams.config["lambda_softmax"], shard_size=128)
    else:
        sims = shard_xattn_t2i(img_embs, cap_embs, stc_lens, pl_module.hparams.config["lambda_softmax"], shard_size=128)
        #sims_0 = shard_xattn_t2i(img_embs_0, cap_embs_0, stc_lens, pl_module.hparams.config["lambda_softmax"], shard_size=128)
        sims_1 = shard_xattn_t2i(img_embs_1, cap_embs_1, stc_lens, pl_module.hparams.config["lambda_softmax"], shard_size=128)
        sims_2 = shard_xattn_t2i(img_embs_2, cap_embs_2, stc_lens, pl_module.hparams.config["lambda_softmax"], shard_size=128)
       

    

    sims_all = sims  + sims_1 + sims_2

    (r1, r5, r10, r20, r50, r70, r100, medr, meanr) = i2t_SCAN(sims)
    (r1i, r5i, r10i, r20i, r50i, r70i, r100i, medri, meanri) = t2i_SCAN(sims)

    print("Image to text: %.1f, %.1f, %.1f, %.1f, %.1f, %.1f, %.1f" % (r1, r5, r10, r20, r50, r70, r100))
        
    print("Text to image: %.1f, %.1f, %.1f, %.1f, %.1f, %.1f, %.1f" % (r1i, r5i, r10i, r20i, r50i, r70i, r100i))

    (r1, r5, r10, r20, r50, r70, r100, medr, meanr) = i2t_SCAN(sims_all)
    (r1i, r5i, r10i, r20i, r50i, r70i, r100i, medri, meanri) = t2i_SCAN(sims_all)

    #np.save('/data2/lihaoxuan/New-Swin-Ablation/test/flickr/runs/sims/i2t.npy', sims_all)
    
    pl_module.log('best_irtr', (r1+r1i))
        

    return (r1, r5, r10, r20, r50, r70, r100, r1i, r5i, r10i, r20i, r50i, r70i, r100i)




@torch.no_grad()
def compute_irtr_recall(pl_module):
    text_dset = pl_module.trainer.datamodule.dms[0].make_no_false_val_dset()
    text_dset.tokenizer = pl_module.trainer.datamodule.dms[0].tokenizer
    text_loader = torch.utils.data.DataLoader(
        text_dset,
        batch_size=64,
        num_workers=pl_module.hparams.config["num_workers"],
        pin_memory=True,
        collate_fn=functools.partial(
            text_dset.collate,
            mlm_collator=pl_module.trainer.datamodule.dms[0].mlm_collator,
        ),
    )

    image_dset = pl_module.trainer.datamodule.dms[0].make_no_false_val_dset(
        image_only=True
    )
    image_dset.tokenizer = pl_module.trainer.datamodule.dms[0].tokenizer
    dist_sampler = DistributedSampler(image_dset, shuffle=False)
    image_loader = torch.utils.data.DataLoader(
        image_dset,
        batch_size=1,
        num_workers=pl_module.hparams.config["num_workers"],
        sampler=dist_sampler,
        pin_memory=True,
        collate_fn=functools.partial(
            image_dset.collate,
            mlm_collator=pl_module.trainer.datamodule.dms[0].mlm_collator,
        ),
    )

    #TODO: speed up the process by caching text/image features
    text_preload = list()
    for _b in tqdm.tqdm(text_loader, desc="text prefetch loop"):
        text_preload.append(
            {
                "text_ids": _b["text_ids"].to(pl_module.device),
                "text_masks": _b["text_masks"].to(pl_module.device),
                "text_labels": _b["text_labels"].to(pl_module.device),
                "img_index": _b["img_index"],
            }
        )

    tiids = list()
    for pre in text_preload:
        tiids += pre["img_index"]
    tiids = torch.tensor(tiids)

    image_preload = list()
    for _b in tqdm.tqdm(image_loader, desc="image prefetch loop"):
        image_preload.append((_b['image'][0], _b["img_index"][0]))

    rank_scores = list()
    rank_iids = list()

    for img_batch in tqdm.tqdm(image_preload, desc="rank loop"):
        _im, _iid = img_batch

        img_batch_score = list()
        for txt_batch in text_preload:
            fblen = len(txt_batch["text_ids"])
            im = _im.repeat(fblen, 1, 1, 1).to(device=txt_batch['text_ids'].device)

            with torch.cuda.amp.autocast():
                score = pl_module.rank_output(
                    pl_module.infer(
                        {
                            "text_ids": txt_batch["text_ids"],
                            "text_masks": txt_batch["text_masks"],
                            "text_labels": txt_batch["text_labels"],
                        },
                        img=im,
                    )["cls_feats"]
                )[:, 0]

            img_batch_score.append(score)

        img_batch_score = torch.cat(img_batch_score)
        rank_scores.append(img_batch_score.cpu().tolist())
        rank_iids.append(_iid)

    torch.distributed.barrier()
    gather_rank_scores = all_gather(rank_scores)
    gather_rank_iids = all_gather(rank_iids)

    iids = torch.tensor(gather_rank_iids)
    iids = iids.view(-1)
    scores = torch.tensor(gather_rank_scores)
    scores = scores.view(len(iids), -1)

    topk10 = scores.topk(10, dim=1)
    topk5 = scores.topk(5, dim=1)
    topk1 = scores.topk(1, dim=1)
    topk10_iids = tiids[topk10.indices]
    topk5_iids = tiids[topk5.indices]
    topk1_iids = tiids[topk1.indices]

    tr_r10 = (iids.unsqueeze(1) == topk10_iids).float().max(dim=1)[0].mean()
    tr_r5 = (iids.unsqueeze(1) == topk5_iids).float().max(dim=1)[0].mean()
    tr_r1 = (iids.unsqueeze(1) == topk1_iids).float().max(dim=1)[0].mean()

    topk10 = scores.topk(10, dim=0)
    topk5 = scores.topk(5, dim=0)
    topk1 = scores.topk(1, dim=0)
    topk10_iids = iids[topk10.indices]
    topk5_iids = iids[topk5.indices]
    topk1_iids = iids[topk1.indices]

    ir_r10 = (tiids.unsqueeze(0) == topk10_iids).float().max(dim=0)[0].mean()
    ir_r5 = (tiids.unsqueeze(0) == topk5_iids).float().max(dim=0)[0].mean()
    ir_r1 = (tiids.unsqueeze(0) == topk1_iids).float().max(dim=0)[0].mean()

    return (ir_r1, ir_r5, ir_r10, tr_r1, tr_r5, tr_r10)


def init_weights(module):
    if isinstance(module, (nn.Linear, nn.Embedding)):
        module.weight.data.normal_(mean=0.0, std=0.02)
    elif isinstance(module, nn.LayerNorm):
        module.bias.data.zero_()
        module.weight.data.fill_(1.0)

    if isinstance(module, nn.Linear) and module.bias is not None:
        module.bias.data.zero_()


def vqa_test_step(pl_module, batch, output):
    try:
        id2answer = (
            pl_module.trainer.datamodule.dm_dicts["vqa_trainval"].id2answer
            if "vqa_trainval" in pl_module.trainer.datamodule.dm_dicts
            else pl_module.trainer.datamodule.dm_dicts["vqa"].id2answer
        )
    except:
        id2answer = (
            pl_module.trainer.datamodule.dm_dicts["gqa_test"].id2answer
            if "gqa_test" in pl_module.trainer.datamodule.dm_dicts
            else pl_module.trainer.datamodule.dm_dicts["gqa"].id2answer
        )
        vqa_logits = output["vqa_logits"]
        vqa_preds = vqa_logits.argmax(dim=-1)
        vqa_preds = [id2answer[pred.item()] for pred in vqa_preds]
        questions = batch["text"]
        qids = batch["qid"]
        return {"qids": qids, "preds": vqa_preds, "gqa": True}
    vqa_logits = output["vqa_logits"]
    vqa_preds = vqa_logits.argmax(dim=-1)
    vqa_preds = [id2answer[pred.item()] for pred in vqa_preds]
    questions = batch["text"]
    qids = batch["qid"]
    return {"qids": qids, "preds": vqa_preds, "gqa": False}


def arc_test_step(pl_module, batch, output):
    return output


def vqa_test_wrapup(outs, model_name):
    rank = torch.distributed.get_rank()
    qids, preds = list(), list()
    gqa = False
    for out in outs:
        qids += out["qids"]
        preds += out["preds"]
        gqa = out['gqa']

    rets = list()
    for qid, pred in zip(qids, preds):
        if gqa:
            rets.append({"questionId": qid, "prediction": pred})
        else:
            rets.append({"question_id": qid, "answer": pred})
    with open(f"vqa_submit_{rank}.json", "w") as fp:
        json.dump(rets, fp, indent=4)

    torch.distributed.barrier()

    if rank == 0:
        jsons = list()
        paths = list(glob.glob("vqa_submit_*.json"))
        for path in paths:
            with open(path, "r") as fp:
                jsons += json.load(fp)
        os.makedirs("result", exist_ok=True)
        with open(f"result/vqa_submit_{model_name}.json", "w") as fp:
            json.dump(jsons, fp, indent=4)

    torch.distributed.barrier()
    os.remove(f"vqa_submit_{rank}.json")


def arc_test_wrapup(outs, caplen, model_name):
    rank = torch.distributed.get_rank()
    iids, captions = list(), list()
    for out in outs:
        iids += out["iid"]
        captions += out["captions"]

    rets = list()
    for iid, caption in zip(iids, captions):
        rets.append({"image_id": iid, "caption": caption})
    with open(f"coco_cap_len{caplen}_{rank}.json", "w") as fp:
        json.dump(rets, fp, indent=4)

    torch.distributed.barrier()

    if rank == 0:
        jsons = list()
        paths = list(glob.glob(f"coco_cap_len{caplen}_*.json"))
        for path in paths:
            with open(path, "r") as fp:
                jsons += json.load(fp)
        os.makedirs("result/arc", exist_ok=True)
        jsons = sorted(jsons, key=lambda x: x["image_id"])
        with open(f"result/arc/coco_cap_{model_name}_len{caplen}.json", "w") as fp:
            json.dump(jsons, fp, indent=4)

    torch.distributed.barrier()
    os.remove(f"coco_cap_len{caplen}_{rank}.json")
