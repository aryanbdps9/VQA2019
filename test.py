import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
import time
import torch
import torch.nn as nn
import utils
from torch.autograd import Variable
import numpy as np
import numpy.random as npr
import random
import torch.optim as optim
import torch.nn.functional as F
from tensorboardX import SummaryWriter
import gc
use_cuda = torch.cuda.is_available()
use_cuda = False
# if use_cuda:
#     torch.set_default_tensor_type('torch.cuda.DoubleTensor')
# else:
#     torch.set_default_tensor_type('torch.DoubleTensor')
device = torch.device('cuda' if use_cuda else 'cpu')
# print("expt device", torch.cuda.get_device_name(device))
# torch.manual_seed(1234)

def get_c2i(char):
    if (char == '\b'):
        return 0
    if (char == '\0'):
        return 1
    return 2 + ord(char)


class Model(nn.Module):
    # c2i_map: 0 for start, 1 for end, rest for rest
    def __init__(self, ndim_word=300, char_vocab=258, emb_size_char=512, rnn_layers=1, rnn_hidden=1024):
        super(Model, self).__init__()
        self.ndim_word, self.char_vocab, self.emb_size_char = ndim_word, char_vocab, emb_size_char
        self.emb = nn.Embedding(char_vocab, emb_size_char)
        self.rnn = nn.LSTM(ndim_word+emb_size_char, rnn_hidden, num_layers=rnn_layers, batch_first=True)
        # self.rnn = myGRU(ndim_word+emb_size_char, nhidden=rnn_hidden)
        # self.rnn = nn.GRU(emb_size_char+ndim_word, hidden_size=rnn_hidden, num_layers=rnn_layers, batch_first=True, dropout=0.5)
        self.lin = nn.Linear(rnn_hidden, char_vocab)
        self.rnn_hidden, self.rnn_layers = rnn_hidden, rnn_layers
        self.softmax = nn.Softmax(dim=1)
        self.lim = 32 # if the new outputs words of len > 32, ditch it

    def forward(self, word_vec):
        bs = word_vec.size(0)
        # torch.cuda.device(device)
        start_tok = self.emb(torch.LongTensor([[0] for _ in range(bs)]).to(device)).view(bs, -1)
        # start_tok = self.emb(torch.LongTensor([[0] for _ in range(bs)])).view(bs, -1)
        # h = self.rnn.init_hidden(bs)
        h = torch.zeros(bs, self.rnn_hidden*self.rnn_layers).view(self.rnn_layers, bs, -1)
        c = torch.zeros(bs, self.rnn_hidden*self.rnn_layers).view(self.rnn_layers, bs, -1)
        resp, resi = [], []
        for i in range(14):
            # print("start_tok size = ", start_tok.size())
            # print("word_vec size = ", word_vec.size())
            xt = torch.cat((start_tok, word_vec.to(device)), dim=1).view(bs, 1, -1)
            # h = self.rnn.forward(xt, h)
            out, (h, c) = self.rnn(xt, (h, c))
            # out = out.view(bs, 1, 1, -1)
            start_tok = self.lin(out).view(bs, -1)
            start_tok = self.softmax(start_tok)
            topk_p, topk_i = torch.topk(start_tok, 1, dim=1, largest=True)
            resp.append(topk_p)
            resi.append(topk_i)
            start_tok = self.emb(topk_i).view(bs, -1)
        return resp, resi

def valid_word(wrd):
    for i in wrd:
        if ord(i) > 255:
            return False
    return True

def print_word(pred_int):
    # print(pred_int)
    listt = []
    for pred in pred_int:
        if (pred != 1 and pred != 0):
            listt.append(chr(pred - 2))
        else:
            break
    return ''.join(listt)
    pred_val = ''.join([chr(pred  - 2) for pred in pred_int if (pred > 1)])
    return pred_val

def load_model(save_path):
    model = Model(rnn_layers=2)
    ckpt = torch.load(save_path)
    model.load_state_dict(ckpt['model_params'])
    del ckpt
    gc.collect()
    return model


def test(model1, dataloader, model_path):
    model2_path = os.path.join(model_path,'model2.pt')
    model2 = load_model(model2_path)
    for v, b, q, a, word_vec, q_type in iter(dataloader):
        # print("In the loop")
        # v = Variable(v, volatile=True).cuda()
        # b = Variable(b, volatile=True).cuda()
        # q = Variable(q, volatile=True).cuda()
        v = Variable(v).cuda()
        b = Variable(b).cuda()
        q = Variable(q).cuda()
        _, pred_word_vec, _, _ = model1(v, b, q, None)
        resp, resi = model2.forward(pred_word_vec)
        try:
            print("Predicted Answer: {}\t\tActual Answer: {}".format(print_word(resi), word_vec))
        except:
            print("Not printing due to unicode issue")
