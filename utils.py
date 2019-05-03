from __future__ import print_function

import errno
import os
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
from tensorboardX import SummaryWriter
import gc
import datetime

emb_dim = 300
GloveEmbeddings = {}
embedding_mat = None
word_to_index_map = {}
juice_path = "data/"
glove_path = os.path.join(juice_path, "glove", "glove.6B.300d.txt")
embedding_path = os.path.join(juice_path, "glove", "embfile.npy")
w2im_path = os.path.join(juice_path, "glove", "w2im.npy")



EPS = 1e-7


def strToNP(inpstr, delimiter=" "):
  listt = inpstr.strip().split(delimiter)
  flist = [float(listitem) for listitem in listt]
  return np.array(flist, dtype='float32')

def loadEmbeddings(embeddingfile):
    global GloveEmbeddings, emb_dim

    fe = open(embeddingfile, "r", encoding="utf-8", errors="ignore")
    for line in fe:
        tokens = line.strip().split()
        word = tokens[0]
        vec = tokens[1:]
        vec = " ".join(vec)
        GloveEmbeddings[word] = vec
    #Add Zerovec, this will be useful to pad zeros, it is better to experiment with padding any non-zero constant values also.
    GloveEmbeddings["zerovec"] = "0.0 " * emb_dim
    GloveEmbeddings["starttoken"] = "0.3 " * emb_dim
    GloveEmbeddings["endtoken"]  = "0.7 " * emb_dim
    fe.close()
    print("loaded Embeddings")

def pop_embmat_w2imap():
    global GloveEmbeddings, emb_dim, embedding_mat, word_to_index_map
    num_words = len(GloveEmbeddings)
    # V X d matrix storing GloVe embeddings
    embedding_mat = np.zeros((num_words, emb_dim), dtype="float32")
    i = 0
    for k in GloveEmbeddings:
        word_to_index_map[k] = i
        embedding_mat[i, :] = strToNP(GloveEmbeddings[k])
        i += 1
    # Can also remove GloveEmbeddings to save RAM
    GloveEmbeddings = {}
    gc.collect()

def get_gloves():
    global embedding_mat, word_to_index_map
    if (os.path.exists(embedding_path)):
        embedding_mat = np.load(embedding_path)
        word_to_index_map = np.load(w2im_path).item()
        print("loaded files from disk")
        return embedding_mat, word_to_index_map
    loadEmbeddings(glove_path)
    pop_embmat_w2imap()
    print("init_embeddings done. saving files to disk")
    np.save(embedding_path, embedding_mat)
    np.save(w2im_path, word_to_index_map)
    return embedding_mat, word_to_index_map


def assert_eq(real, expected):
    assert real == expected, '%s (true) vs %s (expected)' % (real, expected)


def assert_array_eq(real, expected):
    assert (np.abs(real-expected) < EPS).all(), \
        '%s (true) vs %s (expected)' % (real, expected)


def load_folder(folder, suffix):
    imgs = []
    for f in sorted(os.listdir(folder)):
        if f.endswith(suffix):
            imgs.append(os.path.join(folder, f))
    return imgs


def load_imageid(folder):
    images = load_folder(folder, 'jpg')
    img_ids = set()
    for img in images:
        img_id = int(img.split('/')[-1].split('.')[0].split('_')[-1])
        img_ids.add(img_id)
    return img_ids


def pil_loader(path):
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')


def weights_init(m):
    """custom weights initialization."""
    cname = m.__class__
    if cname == nn.Linear or cname == nn.Conv2d or cname == nn.ConvTranspose2d:
        m.weight.data.normal_(0.0, 0.02)
    elif cname == nn.BatchNorm2d:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)
    else:
        print('%s is not initialized.' % cname)


def init_net(net, net_file):
    if net_file:
        net.load_state_dict(torch.load(net_file))
    else:
        net.apply(weights_init)


def create_dir(path):
    if not os.path.exists(path):
        try:
            os.makedirs(path)
        except OSError as exc:
            if exc.errno != errno.EEXIST:
                raise


class Logger(object):
    def __init__(self, output_name):
        dirname = os.path.dirname(output_name)
        self.expt_name = os.path.join(dirname, "run", output_name)
        date_time = str(datetime.datetime.now())
        date_time = date_time.split('.')[0]
        date_time = date_time.replace('-', '_')
        date_time = date_time.replace(':', '_')
        date_time = date_time.replace(' ', '__')
        self.expt_name2 = os.path.join(dirname, "run", date_time)
        print("Logger:expt_name ", self.expt_name2)
        if not os.path.exists(self.expt_name2):
            os.mkdir(self.expt_name2)
        if not os.path.exists(dirname):
            os.mkdir(dirname)
        if not os.path.isdir(self.expt_name):
            os.system('mkdir -p '+self.expt_name)
        self.writer = SummaryWriter(self.expt_name2)
        self.log_file = open(output_name, 'w')
        self.infos = {}

    def append(self, key, val):
        vals = self.infos.setdefault(key, [])
        vals.append(val)

    def log(self, extra_msg=''):
        msgs = [extra_msg]
        for key, vals in self.infos.iteritems():
            msgs.append('%s %.6f' % (key, np.mean(vals)))
        msg = '\n'.join(msgs)
        self.log_file.write(msg + '\n')
        self.log_file.flush()
        self.infos = {}
        return msg

    def write(self, msg):
        self.log_file.write(msg + '\n')
        self.log_file.flush()
        print(msg)
