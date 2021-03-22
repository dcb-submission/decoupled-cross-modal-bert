import torch
import torch.nn as nn
import torch.nn.init
import torchvision.models as models
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.nn.utils.weight_norm import weight_norm
import torch.backends.cudnn as cudnn
from torch.nn.utils.clip_grad import clip_grad_norm
import numpy as np
from collections import OrderedDict
import torch.nn.functional as F
from modeling_bertnew import BertModelNew
from apex import amp
import random

def compute_loss(scores,margin=0.2):

    diagonal = scores.diag().view(scores.size(0), 1)
    d1 = diagonal.expand_as(scores)
    d2 = diagonal.t().expand_as(scores)
    cost_s = (margin + scores - d1).clamp(min=0)
    cost_im = (margin + scores - d2).clamp(min=0)
    mask = torch.eye(scores.size(0)) > .5
    I = Variable(mask).cuda()
    cost_s = cost_s.masked_fill_(I, 0)
    cost_im = cost_im.masked_fill_(I, 0)
    eps = 1e-5
    cost_s = cost_s.pow(8).sum(1).add(eps).sqrt().sqrt().sqrt()#.sqrt()#.div(cost_s.size(1)).mul(2)
    cost_im = cost_im.pow(8).sum(0).add(eps).sqrt().sqrt().sqrt()#.sqrt()#.div(cost_im.size(0)).mul(2)
    return cost_s.sum().div(cost_s.size(0)) + cost_im.sum().div(cost_s.size(0))


def get_non_pad_mask(seq):
    assert seq.dim() == 2
    return seq.ne(0).type(torch.float).unsqueeze(-1)


def l2norm(X, dim, eps=1e-5):
    """L2-normalize columns of X
    """
    norm = torch.pow(X, 2).sum(dim=dim, keepdim=True).add(eps).sqrt() + eps
    X = torch.div(X, norm)
    return X



# RNN Based Language Model





class Encoder(nn.Module):

    def __init__(self,img_dim, embed_size):
        super(Encoder, self).__init__()
        self.encoder = BertModelNew.from_pretrained('bert/')
        self.fc = nn.Linear(img_dim,embed_size)

    def forward(self, input_ids, token_type_ids, non_pad_mask, vision_feat, vision_mask, istest=False):
        text_output = self.encoder.embeddings(input_ids=input_ids, position_ids=None, token_type_ids=token_type_ids.long().squeeze())
        text_output = text_output.mul(non_pad_mask.unsqueeze(2).expand(text_output.size()))  

        text_g = text_output.sum(1)

        head_mask = [None]*20
        vision_feat = self.fc(vision_feat)
        vision_feat = self.encoder.embeddings.LayerNorm(vision_feat)

        word_ids = torch.arange(30522, dtype=torch.long).cuda()
        word_all = self.encoder.embeddings.word_embeddings(word_ids)
        word_all = self.encoder.embeddings.LayerNorm(word_all)
        word_all = word_all.permute(1,0)


        vision_feat_new = vision_feat
        vision_g = vision_feat_new.sum(1)


        scores = torch.matmul(vision_feat_new,word_all).mul(20)
        scores = F.softmax(scores,2)
        featnew = torch.matmul(scores,word_all.permute(1,0))
        vision_feat = torch.cat([vision_feat_new,featnew],1)
        vision_mask = torch.cat([vision_mask,vision_mask],1)

        bs = text_output.size(0)
        tl = text_output.size(1)
        vl = vision_feat.size(1)

        extended_attention_mask_text = non_pad_mask[:, None, None, :]
        extended_attention_mask_text = (1.0 - extended_attention_mask_text) * -10000.0

        extended_attention_mask_vision = vision_mask[:, None, None, :]
        extended_attention_mask_vision = (1.0 - extended_attention_mask_vision) * -10000.0


        textnew = self.encoder.encoder(text_output,extended_attention_mask_text,head_mask)
        visionnew = self.encoder.encoder(vision_feat,extended_attention_mask_vision,head_mask) 

        textnew = textnew[0]
        visionnew = visionnew[0]

        text_out = textnew[:,0]
        vision_output =  visionnew.sum(1)

        if istest == False:
            text_out = text_out.unsqueeze(0).expand(bs,-1,-1).contiguous().view(bs*bs,-1)#.view(bs,bs,-1)
            vision_output = vision_output.unsqueeze(1).expand(-1,bs,-1).contiguous().view(bs*bs,-1)
            text_g = text_g.unsqueeze(0).expand(bs,-1,-1).contiguous().view(bs*bs,-1)#.view(bs,bs,-1)
            vision_g = vision_g.unsqueeze(1).expand(-1,bs,-1).contiguous().view(bs*bs,-1)
        else:
            return  vision_output,text_out,vision_g,text_g

        scores =  cosine_similarity(vision_output,text_out,-1)
        scores_g =  cosine_similarity(vision_g,text_g,-1)

        if istest:  
            return scores + scores_g#*1
        else:
            scores = scores.view(bs,bs)
            scores_g = scores_g.view(bs,bs)
            return compute_loss(scores) + compute_loss(scores_g)



def cosine_similarity(x1, x2, dim=1, eps=1e-5):
    """Returns cosine similarity between x1 and x2, computed along dim."""
    w12 = torch.sum(x1 * x2, dim)
    w1 = torch.norm(x1, 2, dim)
    w2 = torch.norm(x2, 2, dim)
    return (w12 / (w1 * w2).clamp(min=eps)).squeeze()






class DCB(object):
    """
    Stacked Cross Attention Network (SCAN) model
    """
    def __init__(self, opt):
        # Build Models
        self.grad_clip = opt.grad_clip
        self.enc = Encoder(2048,768)
        self.enc.cuda()
        cudnn.benchmark = True

        params = list(self.enc.parameters())
        self.params = params
        self.optimizer = torch.optim.Adam(params, lr=opt.learning_rate)
        self.enc, self.optimizer = amp.initialize(self.enc, self.optimizer, opt_level= "O1")
        self.enc = torch.nn.DataParallel(self.enc)
        # Loss and Optimizer
        self.Eiters = 0

    def state_dict(self):
        state_dict = [self.enc.state_dict()]
        return state_dict

    def load_state_dict(self, state_dict):
        self.enc.load_state_dict(state_dict[0])

    def train_start(self):
        """switch to train mode
        """
        self.enc.train()

    def val_start(self):
        """switch to evaluate mode
        """
        self.enc.eval()

    def forward_emb(self, images, captions,  target_mask, vision_mask, volatile=False, istest = False):
        """Compute the image and caption embeddings
        """
        # Set mini-batch dataset
        images = Variable(images.float(), volatile=volatile)
        captions = torch.LongTensor(captions)
        captions = Variable(captions, volatile=volatile)
        if torch.cuda.is_available():
            images = images.cuda()#.cuda()
            captions = captions.cuda()#.cuda()
        # Forward
        n_img = images.size(0)
        n_cap = captions.size(0)

        attention_mask = get_non_pad_mask(captions).cuda().squeeze()
        token_type_ids = torch.zeros_like(attention_mask)

        video_non_pad_mask = get_non_pad_mask(vision_mask).cuda().squeeze()

        scores = self.enc(captions, token_type_ids, attention_mask,images,video_non_pad_mask,istest)
        return scores

    def train_emb(self, images, captions, target_mask, vision_mask, ids=None, *args):
        """One training step given images and captions.
        """
        self.Eiters += 1
        self.logger.update('Eit', self.Eiters)
        self.logger.update('lr', self.optimizer.param_groups[0]['lr'])
         
        # measure accuracy and record loss
        scores = self.forward_emb(images, captions, target_mask, vision_mask)
        # measure accuracy and record loss

        self.optimizer.zero_grad()
        if scores is not None:
           loss = scores.sum()
           self.logger.update('Le', loss, images.size(0))
        else:
           return
        # compute gradient and do SGD step
        #loss.backward()
        with amp.scale_loss(loss, self.optimizer) as scaled_loss:
           scaled_loss.backward()
        if self.grad_clip > 0:
            clip_grad_norm(self.params, self.grad_clip)
        self.optimizer.step()



