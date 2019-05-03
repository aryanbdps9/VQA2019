import os
os.environ['CUDA_VISIBLE_DEVICES']='0'
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from dataset import Dictionary, VQAFeatureDataset
import base_model
from train import train
import utils


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--num_hid', type=int, default=1024)
    parser.add_argument('--model', type=str, default='baseline0_newatt')
    parser.add_argument('--output', type=str, default='saved_models/exp0')
    parser.add_argument('--batch_size', type=int, default=512)
    parser.add_argument('--seed', type=int, default=0, help='random seed')
    args = parser.parse_args()
    return args

def myfunc():
    args = parse_args()

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.backends.cudnn.benchmark = True

    dictionary = Dictionary.load_from_file('data/dictionary.pkl')
    train_dset = VQAFeatureDataset('train', dictionary)
    return train_dset

if __name__ == '__main__' and True:
    args = parse_args()

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.backends.cudnn.benchmark = True
    emb_mat, w2imap = utils.get_gloves()

    dictionary = Dictionary.load_from_file('data/dictionary.pkl')
    train_dset = VQAFeatureDataset('train', dictionary=dictionary, glove_arr=emb_mat, w2glov=w2imap, dataroot='data')
    eval_dset = VQAFeatureDataset('val', dictionary=dictionary, glove_arr=emb_mat, w2glov=w2imap)
    batch_size = args.batch_size

    train_loader = DataLoader(train_dset, batch_size, shuffle=True, num_workers=8)
    eval_loader =  DataLoader(eval_dset, batch_size, shuffle=True, num_workers=8)
    # train_dset.__getitem__(2)

    constructor = 'build_%s' % args.model
    model = getattr(base_model, constructor)(train_dset, args.num_hid).cuda()
    model.w_emb.init_embedding('data/glove6b_init_300d.npy')
    model.cuda()

    model = nn.DataParallel(model).cuda()
    train(model, train_loader, eval_loader, args.epochs, args.output)
