import torch
import torch.nn as nn
from attention import Attention, NewAttention
from language_model import WordEmbedding, QuestionEmbedding
from classifier import SimpleClassifier
from fc import FCNet

class YNclassifier(nn.Module):
    def __init__(self, ndim=1024, n_hid=1024, dropout=0.0):
        super(YNclassifier, self).__init__()
        self.ndim = ndim
        self.n_hid = n_hid
        self.dropout = dropout
        self.rnn = QuestionEmbedding(ndim, n_hid, 1, False, dropout)
        self.activ = nn.LeakyReLU()
        self.Lin = nn.Sequential(
            nn.Linear(n_hid, 512),
            self.activ,
            nn.Linear(512, 128),
            self.activ,
            nn.Linear(128,3)
        )
    
    def forward(self, q):
        rnn_out = self.rnn(q)
        out = self.Lin(rnn_out)
        return out

class BaseModel(nn.Module):
    def __init__(self, w_emb, q_emb, v_att, q_net, v_net, classifier, dropout=0.0):
        super(BaseModel, self).__init__()
        self.w_emb = w_emb
        self.q_emb = q_emb
        self.v_att = v_att
        self.q_net = q_net
        self.v_net = v_net
        self.classifier = classifier
        self.funnel = nn.Linear(1024, 300)
        self.q_classifier = YNclassifier(ndim=1024, n_hid=1024, dropout=dropout)

    def forward(self, v, b, q, labels):
        """Forward

        v: [batch, num_objs, obj_dim]
        b: [batch, num_objs, b_dim]
        q: [batch_size, seq_length]

        return: logits, not probs
        """
        # w_emb = self.w_emb(q)
        w_emb = q
        qtype = self.q_classifier(q)
        q_emb = self.q_emb(w_emb) # [batch, q_dim]
        # q_emb = q
        att = self.v_att(v, q_emb)
        v_emb = (att * v).sum(1) # [batch, v_dim]

        q_repr = self.q_net(q_emb)
        v_repr = self.v_net(v_emb)
        joint_repr = q_repr * v_repr
        logits = self.classifier(joint_repr)
        return logits, self.funnel(joint_repr), qtype

def build_baseline0(dataset, num_hid):
    w_emb = WordEmbedding(dataset.dictionary.ntoken, 300, 0.0)
    q_emb = QuestionEmbedding(300, num_hid, 1, False, 0.0)
    v_att = Attention(dataset.v_dim, q_emb.num_hid, num_hid)
    q_net = FCNet([num_hid, num_hid])
    v_net = FCNet([dataset.v_dim, num_hid])
    classifier = SimpleClassifier(
        num_hid, 2 * num_hid, dataset.num_ans_candidates, 0.5)
    return BaseModel(w_emb, q_emb, v_att, q_net, v_net, classifier)


def build_baseline0_newatt(dataset, num_hid):
    w_emb = WordEmbedding(dataset.dictionary.ntoken, 300, 0.0)
    q_emb = QuestionEmbedding(1024, num_hid, 1, False, 0.0)
    v_att = NewAttention(dataset.v_dim, q_emb.num_hid, num_hid)
    q_net = FCNet([q_emb.num_hid, num_hid])
    v_net = FCNet([dataset.v_dim, num_hid])
    classifier = SimpleClassifier(
        num_hid, num_hid * 2, dataset.num_ans_candidates, 0.5)
    return BaseModel(w_emb, q_emb, v_att, q_net, v_net, classifier)
