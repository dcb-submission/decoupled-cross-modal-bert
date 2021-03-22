"""Evaluation"""

from __future__ import print_function
import os

import sys
import time
import numpy as np
import torch
from model import DCB
from collections import OrderedDict
import time
from torch.autograd import Variable


def cosine_similarity(x1, x2, dim=1, eps=1e-5):
    """Returns cosine similarity between x1 and x2, computed along dim."""
    w12 = torch.sum(x1 * x2, dim)
    w1 = torch.norm(x1, 2, dim)
    w2 = torch.norm(x2, 2, dim)
    return (w12 / (w1 * w2).clamp(min=eps)).squeeze()

def get_non_pad_mask(seq):
    assert seq.dim() == 2
    return seq.ne(0).type(torch.float).unsqueeze(-1)

def get_attn_key_pad_mask(seq_k, seq_q):
    ''' For masking out the padding part of key sequence. '''
    # Expand to fit the shape of key query attention matrix.
    if seq_q.dim() == 1:
        seq_q = seq_q.unsqueeze(0)
    if seq_k.dim() == 1:
        seq_k = seq_k.unsqueeze(0)
    len_q = seq_q.size(1)
    #padding_mask = seq_k.eq(Constants.PAD)
    padding_mask = seq_k.eq(0)
    padding_mask = padding_mask.unsqueeze(1).expand(-1, len_q, -1)  # b x lq x lk
    return padding_mask


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=0):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / (.0001 + self.count)

    def __str__(self):
        """String representation for logging
        """
        # for values that should be recorded exactly e.g. iteration number
        if self.count == 0:
            return str(self.val)
        # for stats
        return '%.4f (%.4f)' % (self.val, self.avg)


class LogCollector(object):
    """A collection of logging objects that can change from train to val"""

    def __init__(self):
        # to keep the order of logged variables deterministic
        self.meters = OrderedDict()

    def update(self, k, v, n=0):
        # create a new meter if previously not recorded
        if k not in self.meters:
            self.meters[k] = AverageMeter()
        self.meters[k].update(v, n)

    def __str__(self):
        """Concatenate the meters in one log line
        """
        s = ''
        for i, (k, v) in enumerate(self.meters.items()):
            if i > 0:
                s += '  '
            s += k + ' ' + str(v)
        return s

    def tb_log(self, tb_logger, prefix='', step=None):
        """Log using tensorboard
        """
        for k, v in self.meters.items():
            tb_logger.log_value(prefix + k, v.val, step=step)


def encode_data(model, data_loader, log_step=10, logging=print):
    """Encode all images and captions loadable by `data_loader`
    """
    batch_time = AverageMeter()
    val_logger = LogCollector()

    # switch to evaluate mode
    model.val_start()

    end = time.time()

    # np array to keep all the embeddings
    img_embs = None
    cap_embs = None
    cap_lens = None
    text_pads = None
    text_selfs = None
    img_pads = None
    img_selfs = None

    max_n_word = 0

    for i, (images, captions, target_mask, vision_mask, ids) in enumerate(data_loader):
        # make sure val logger is used
        model.logger = val_logger
        
        if img_embs is None:
            img_embs = np.zeros((len(data_loader.dataset), 768))
            cap_embs = np.zeros((len(data_loader.dataset), 768))
            img_embs_g = np.zeros((len(data_loader.dataset), 768))
            cap_embs_g = np.zeros((len(data_loader.dataset), 768))

        attention_mask = get_non_pad_mask(captions).cuda().squeeze()
        token_type_ids = torch.zeros_like(attention_mask)
        video_non_pad_mask = get_non_pad_mask(vision_mask).cuda().squeeze()

        img_tmp,cap_tmp,img_g,cap_g = model.enc(captions, token_type_ids, attention_mask,images,video_non_pad_mask,True)
        img_embs[ids] = img_tmp.data.cpu().numpy().copy()
        cap_embs[ids] = cap_tmp.data.cpu().numpy().copy()
        img_embs_g[ids] = img_g.data.cpu().numpy().copy()
        cap_embs_g[ids] = cap_g.data.cpu().numpy().copy()

    return img_embs, cap_embs,img_embs_g, cap_embs_g




def t2i_model(model, images, captions, images_g, captions_g,  opt, shard_size=128):
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
            n_cap = cap_end - cap_start
            n_img = im_end - im_start

            im = Variable(torch.from_numpy(images[im_start:im_end])).float().cuda()#Variable(torch.from_numpy(images[im_start:im_end]), volatile=True).float().cuda()
            cap = Variable(torch.from_numpy(captions[cap_start:cap_end])).float().cuda()#Variable(torch.from_numpy(captions[cap_start:cap_end]), volatile=True).float().cuda()

            im = im.unsqueeze(1).expand(n_img,n_cap,im.size(1)).contiguous().view(-1,im.size(1))
            cap = cap.unsqueeze(0).expand(n_img,n_cap,cap.size(1)).contiguous().view(-1,cap.size(1))#.contiguous().

            scores = cosine_similarity(im,cap)#model.txt_enc(cap, cap_type, cap_mask,im,img_mask,istest=True)#.matmul(im,s)
            sim = scores.view(n_img,n_cap)

            im_g = Variable(torch.from_numpy(images_g[im_start:im_end])).float().cuda()#Variable(torch.from_numpy(images[im_start:im_end]), volatile=True).float().cuda()
            cap_g = Variable(torch.from_numpy(captions_g[cap_start:cap_end])).float().cuda()#Variable(torch.from_numpy(captions[cap_start:cap_end]), volatile=True).float().cuda()

            im_g = im_g.unsqueeze(1).expand(n_img,n_cap,im.size(1)).contiguous().view(-1,im.size(1))
            cap_g = cap_g.unsqueeze(0).expand(n_img,n_cap,cap.size(1)).contiguous().view(-1,cap.size(1))#.contiguous().

            scores_g = cosine_similarity(im_g,cap_g)#model.txt_enc(cap, cap_type, cap_mask,im,img_mask,istest=True)#.matmul(im,s)
            sim_g = scores_g.view(n_img,n_cap)

            sim = sim + sim_g

 
            d[im_start:im_end, cap_start:cap_end] = sim.data.cpu().numpy()
    sys.stdout.write('\n')
    return d


def i2t_model(model, images, captions, opt, shard_size=128):
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
            n_cap = cap_end - cap_start
            n_img = im_end - im_start

            im = Variable(torch.from_numpy(images[im_start:im_end])).float().cuda()#Variable(torch.from_numpy(images[im_start:im_end]), volatile=True).float().cuda()
            cap = Variable(torch.from_numpy(captions[cap_start:cap_end])).float().cuda()#Variable(torch.from_numpy(captions[cap_start:cap_end]), volatile=True).float().cuda()

            im = im.unsqueeze(1).expand(n_img,n_cap,im.size(1)).contiguous().view(-1,im.size(1))
            cap = cap.unsqueeze(0).expand(n_img,n_cap,cap.size(1)).contiguous().view(-1,cap.size(1))#.contiguous().
            scores = cosine_similarity(im,cap)#model.txt_enc(cap, cap_type, cap_mask,im,img_mask,istest=True)#.matmul(im,s)
            sim = scores.view(n_img,n_cap)#model.cross_att(img_emb,cap_emb,test=True)
            d[im_start:im_end, cap_start:cap_end] = sim.data.cpu().numpy()
    sys.stdout.write('\n')
    return d

def i2t(images, captions, sims, npts=None, return_ranks=False):
    """
    Images->Text (Image Annotation)
    Images: (N, n_region, d) matrix of images
    Captions: (5N, max_n_word, d) matrix of captions
    CapLens: (5N) array of caption lengths
    sims: (N, 5N) matrix of similarity im-cap
    """
    npts = images.shape[0]
    ranks = np.zeros(npts)
    top1 = np.zeros(npts)
    for index in range(npts):
        inds = np.argsort(sims[index])[::-1]
        # Score
        rank = 1e20
        for i in range(5 * index, 5 * index + 5, 1):
            tmp = np.where(inds == i)[0][0]
            if tmp < rank:
                rank = tmp
        ranks[index] = rank
        top1[index] = inds[0]

    # Compute metrics
    r1 = 100.0 * len(np.where(ranks < 1)[0]) / len(ranks)
    r5 = 100.0 * len(np.where(ranks < 5)[0]) / len(ranks)
    r10 = 100.0 * len(np.where(ranks < 10)[0]) / len(ranks)
    medr = np.floor(np.median(ranks)) + 1
    meanr = ranks.mean() + 1
    if return_ranks:
        return (r1, r5, r10, medr, meanr), (ranks, top1)
    else:
        return (r1, r5, r10, medr, meanr)


def t2i(images, captions, sims, npts=None, return_ranks=False):
    """
    Text->Images (Image Search)
    Images: (N, n_region, d) matrix of images
    Captions: (5N, max_n_word, d) matrix of captions
    CapLens: (5N) array of caption lengths
    sims: (N, 5N) matrix of similarity im-cap
    """
    npts = images.shape[0]
    ranks = np.zeros(5 * npts)
    top1 = np.zeros(5 * npts)

    # --> (5N(caption), N(image))
    sims = sims.T

    for index in range(npts):
        for i in range(5):
            inds = np.argsort(sims[5 * index + i])[::-1]
            ranks[5 * index + i] = np.where(inds == index)[0][0]
            top1[5 * index + i] = inds[0]

    # Compute metrics
    r1 = 100.0 * len(np.where(ranks < 1)[0]) / len(ranks)
    r5 = 100.0 * len(np.where(ranks < 5)[0]) / len(ranks)
    r10 = 100.0 * len(np.where(ranks < 10)[0]) / len(ranks)
    medr = np.floor(np.median(ranks)) + 1
    meanr = ranks.mean() + 1
    if return_ranks:
        return (r1, r5, r10, medr, meanr), (ranks, top1)
    else:
        return (r1, r5, r10, medr, meanr)

