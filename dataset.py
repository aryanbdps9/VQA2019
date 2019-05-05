from __future__ import print_function
import os
import json
import cPickle
# import _pickle as cPickle
import numpy as np
import utils
# import torch.multiprocessing
# torch.multiprocessing.set_start_method('spawn', force=True)
import h5py
import torch
from torch.utils.data import Dataset


class Dictionary(object):
    def __init__(self, word2idx=None, idx2word=None):
        if word2idx is None:
            word2idx = {}
        if idx2word is None:
            idx2word = []
        self.word2idx = word2idx
        self.idx2word = idx2word

    @property
    def ntoken(self):
        return len(self.word2idx)

    @property
    def padding_idx(self):
        return len(self.word2idx)

    def tokenize(self, sentence, add_word):
        sentence = sentence.lower()
        sentence = sentence.replace(',', '').replace('?', '').replace('\'s', ' \'s')
        words = sentence.split()
        tokens = []
        if add_word:
            for w in words:
                tokens.append(self.add_word(w))
        else:
            for w in words:
                tokens.append(self.word2idx[w])
        return tokens

    def dump_to_file(self, path):
        cPickle.dump([self.word2idx, self.idx2word], open(path, 'wb'))
        print('dictionary dumped to %s' % path)

    @classmethod
    def load_from_file(cls, path):
        print('loading dictionary from %s' % path)
        word2idx, idx2word = cPickle.load(open(path, 'rb'))
        d = cls(word2idx, idx2word)
        return d

    def add_word(self, word):
        if word not in self.word2idx:
            self.idx2word.append(word)
            self.word2idx[word] = len(self.idx2word) - 1
        return self.word2idx[word]

    def __len__(self):
        return len(self.idx2word)


def _create_entry(img, question, answer):
    answer.pop('image_id')
    answer.pop('question_id')
    entry = {
        'question_id' : question['question_id'],
        'image_id'    : question['image_id'],
        'image'       : img,
        'question'    : question['question'],
        'answer'      : answer}
    return entry

def save_dict_to_file(dic, path):
    fn = os.path.join(path, 'dict.txt')
    f = open(fn,'w')
    f.write(str(dic))
    f.close()

def load_dict_from_file(path):
    fn = os.path.join(path, 'dict.txt')
    f = open(fn,'r')
    data=f.read()
    f.close()
    return eval(data)

def _load_dataset(dataroot, name, img_id2val):
    """Load entries

    img_id2val: dict {img_id -> val} val can be used to retrieve image or features
    dataroot: root path of dataset
    name: 'train', 'val'
    """
    question_path = os.path.join(
        dataroot, 'v2_OpenEnded_mscoco_%s2014_questions.json' % name)
    questions = sorted(json.load(open(question_path))['questions'],
                       key=lambda x: x['question_id'])
    answer_path = os.path.join(dataroot, 'cache', '%s_target.pkl' % name)
    answers = cPickle.load(open(answer_path, 'rb'))
    answers = sorted(answers, key=lambda x: x['question_id'])

    utils.assert_eq(len(questions), len(answers))
    entries = []
    for question, answer in zip(questions, answers):
        utils.assert_eq(question['question_id'], answer['question_id'])
        utils.assert_eq(question['image_id'], answer['image_id'])
        img_id = question['image_id']
        entries.append(_create_entry(img_id2val[img_id], question, answer))

    return entries


class VQAFeatureDataset(Dataset):
    def __init__(self, name, dictionary, dataroot='data', glove_arr=None, w2glov=None):
        super(VQAFeatureDataset, self).__init__()
        assert name in ['train', 'val']
        print("dataroot", dataroot)
        self.name = name
        ans2label_path = os.path.join(dataroot, 'cache', 'trainval_ans2label.pkl')
        label2ans_path = os.path.join(dataroot, 'cache', 'trainval_label2ans.pkl')
        # print("Loading ans2label...")
        self.ans2label = cPickle.load(open(ans2label_path, 'rb'))
        # print("Loading label2ans...")
        self.label2ans = cPickle.load(open(label2ans_path, 'rb'))
        self.glove_arr, self.w2glov = glove_arr, w2glov
        self.label2glove = {self.ans2label[w]:w2glov.get(w, -3) for w in self.ans2label.keys()}
        # ans_path = os.path.join(dataroot, 'glove')
        # ans2glove = {w:self.glove_arr[w2glov.get(w, -3),:] for w in self.ans2label.keys()}
        # save_dict_to_file(ans2glove, ans_path)
        # best_label_repr = torch.from_numpy(self.glove_arr[self.label2glove[best_label],:])
        # self.label2glove = {ans2label[w]:w2glov[w] for w in self.ans2label.keys()}
        self.num_ans_candidates = len(self.ans2label)

        self.dictionary = dictionary
        self.foLimit = 4

        # print("Loading imgid2idx..." % name)
        self.img_id2idx = cPickle.load(
            open(os.path.join(dataroot, '%s36_imgid2idx.pkl' % name)))
        print('loading features from h5 file')
        h5_path = os.path.join(dataroot, '%s36.hdf5' % name)
        self.h5_path = h5_path
        print("h5_path", h5_path)
        hf = h5py.File(h5_path, 'r', swmr=True)

            # print("Creating array for features and spatials")
            # self.features = np.array(hf.get('image_features'))
            # self.spatials = np.array(hf.get('spatial_features'))
        self.features = hf['image_features']
        self.spatials = hf['spatial_features']

        question_idx_path = os.path.join(dataroot, '%s_questions_idx.txt' % name)
        f_question_idx = open(question_idx_path, 'r')
        # questions_idx = hf_question_idx['idx']
        # for i in range(questions_idx.size):

        q_idx = f_question_idx.read().strip().split()
        self.questions_idx = {int(q_idx[i]):str(i) for i in range(len(q_idx))}

        question_elmo_path = os.path.join(dataroot, '%s_questions_elmo.hdf5' % name)
        self.question_elmo_path = question_elmo_path
        hf_question_elmo = h5py.File(question_elmo_path, 'r', swmr=True)
        print("hf_question_elmo.swmr_mode", hf_question_elmo.swmr_mode)
        self.questions_elmo = hf_question_elmo #['elmo']

        self.entries = _load_dataset(dataroot, name, self.img_id2idx)

        self.tokenize()
        self.tensorize()
        # print("type features = ", type(self.features))#, type(self.features[0]))
        self.v_dim = self.features[0:1].shape[2]
        self.s_dim = self.spatials[0:1].shape[2]
        hf.close()
        hf_question_elmo.close()

    def tokenize(self, max_length=14):
        """Tokenizes the questions.

        This will add q_token in each entry of the dataset.
        -1 represent nil, and should be treated as padding_idx in embedding
        """
        for entry in self.entries:
            tokens = self.dictionary.tokenize(entry['question'], False)
            tokens = tokens[:max_length]
            if len(tokens) < max_length:
                # Note here we pad in front of the sentence
                padding = [self.dictionary.padding_idx] * (max_length - len(tokens))
                tokens = padding + tokens
            utils.assert_eq(len(tokens), max_length)
            entry['q_token'] = tokens

    def tensorize(self):
        # self.features = torch.from_numpy(self.features)
        # self.spatials = torch.from_numpy(self.spatials)

        for entry in self.entries:
            question = torch.from_numpy(np.array(entry['q_token']))
            entry['q_token'] = question

            answer = entry['answer']
            labels = np.array(answer['labels'])
            scores = np.array(answer['scores'], dtype=np.float32)
            if len(labels):
                best_label = labels[np.argmax(scores)]
                best_label_repr = torch.from_numpy(self.glove_arr[self.label2glove[best_label],:])
                labels = torch.from_numpy(labels)
                scores = torch.from_numpy(scores)
                entry['answer']['labels'] = labels
                entry['answer']['scores'] = scores
                entry['answer']['best_ansvec'] = best_label_repr
            else:
                entry['answer']['labels'] = None
                entry['answer']['scores'] = None
                entry['answer']['best_ansvec'] = None

    def __getitem__(self, index):
        # index = 247573
        # print("getitem:{}[{}]".format(self.name, index))
        entry = self.entries[index]
        qid = entry['question_id']
        # print(type(qid))
        # print("entry['image']", entry['image'], "index", index)

        # features = torch.from_numpy(self.features[entry['image']])
        # # print("features size = ", features.size())
        # spatials = torch.from_numpy(self.spatials[entry['image']])
        # question = torch.tensor(self.questions_elmo[self.questions_idx[qid]])

        if(self.foLimit > 0):
            self.foLimit -= 1
            self.hflater = h5py.File(self.h5_path, 'r', swmr=True)
            features = torch.from_numpy(self.hflater['image_features'][entry['image']])
            spatials = torch.from_numpy(self.hflater['spatial_features'][entry['image']])
            self.qhflater = h5py.File(self.question_elmo_path, 'r', swmr=True)
            question = torch.tensor(self.qhflater[self.questions_idx[qid]])
        else:
            features = torch.from_numpy(self.hflater['image_features'][entry['image']])
            spatials = torch.from_numpy(self.hflater['spatial_features'][entry['image']])
            question = torch.tensor(self.qhflater[self.questions_idx[qid]])

        # with h5py.File(self.h5_path, 'r', swmr=True) as hf:
        #     features = torch.from_numpy(hf['image_features'][entry['image']])
        #     # print("features size = ", features.size())
        #     spatials = torch.from_numpy(hf['spatial_features'][entry['image']])
        # with h5py.File(self.question_elmo_path, 'r', swmr=True) as qhf:
        #     question = torch.tensor(qhf[self.questions_idx[qid]])
        qs = question.size(0)
        if qs > 14:
            question = question[:14, :]
        elif qs < 14:
            q = torch.zeros(14, 1024)
            q[:qs, :] = question
            question = q

        # question = entry['q_token']

        # question = torch.tensor(self.questions_elmo[self.questions_idx[qid]])
        answer = entry['answer']
        labels = answer['labels']
        scores = answer['scores']
        ans_vec = answer['best_ansvec']
        if ans_vec is None:
            ans_vec = torch.zeros(300)
            # print("ans_vecof bestansvec is None")
        target = torch.zeros(self.num_ans_candidates)
        if labels is not None:
            target.scatter_(0, labels, scores)

        # print("getitem:{}[{}]:\tquestion[{}], ans_vec[{}]".format(self.name, index,question.size(), ans_vec.size()))
        return features, spatials, question, target, ans_vec

    def __len__(self):
        return len(self.entries)
