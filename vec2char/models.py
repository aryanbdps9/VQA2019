import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
import numpy as np
import numpy.random as npr
import random
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from tensorboardX import SummaryWriter
import gc
from Queue import PriorityQueue
from embedding import get_c2i
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("expt device", torch.cuda.get_device_name(device))
torch.cuda.device(device)
torch.manual_seed(5042)
torch.set_default_tensor_type('torch.cuda.FloatTensor')

exp_tb_path_base = "run/exp4/"
exp_tb_path=""
writer = None
save_path = "weights4/train.pt"
save_path2 = "weights4/train_100.pt"

w2vc_path = "data/glove/answer2glove.npy"

# class myGRU(nn.Module):
# 	def __init__(self, ndim=128, nhidden=1024):
# 		super(myGRU, self).__init__()
# 		self.Wir = nn.Linear(ndim, nhidden)
# 		self.Whr = nn.Linear(nhidden, nhidden)
# 		self.Wiz = nn.Linear(ndim, nhidden)
# 		self.Whz = nn.Linear(nhidden, nhidden)
# 		self.Win = nn.Linear(ndim, nhidden)
# 		self.Whn = nn.Linear(nhidden, nhidden)
# 		self.lin = nn.Linear(nhidden, ndim)
# 		self.sigmoid = nn.Sigmoid()
# 		self.tanh = nn.Tanh()
# 		self.ndim, self.nhidden = ndim, nhidden

# 	def forward(self, xt, htm1):
# 		sigmoid, tanh = self.sigmoid, self.tanh
# 		rt = sigmoid(self.Wir(xt)+self.Whr(htm1))
# 		zt = sigmoid(self.Wiz(xt)+self.Whz(htm1))
# 		ht_tilda = tanh(self.Win(xt)+self.Whn(htm1)*rt)
# 		ht = (1.0 - zt)*ht_tilda + zt*htm1
# 		return ht
# 	def init_hidden(self, bs):
# 		return torch.zeros(bs, self.nhidden)

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
	# def forward2(self, word_vec, arr, c2i_map=None, tfratio=1, beam_width=5):
	# 	endval = PriorityQueue()
	# 	nodes = PriorityQueue()
	# 	nodes.put((-1,[[0,torch.tensor(1.0)]]))
	# 	for _ in range(beam_width * self.lim) :
	# 		sc,lok = nodes.get()
	# 		if len(lok) > self.lim:
	# 			continue
	# 		use_teacher_forcing = True if random.random() < tfratio else False
	# 		if use_teacher_forcing and len(lok) <= len(arr) and c2i_map is not None:
	# 			lok[-1][0] = c2i_map(arr[len(lok)-1])
	# 			lok[-1][1] = torch.tensor(1.0)
	# 		if lok[-1][0] < 0 or lok[-1][0] > 257:
	# 			print("lok-10", lok[-1][0])
	# 		tok = self.emb(torch.LongTensor([[lok[-1][0]]]))
	# 		rnn_out = self.rnn(torch.cat((word_vec, tok), dim=2))[1]
	# 		out_gru = self.softmax(self.lin(rnn_out)).view(-1)
	# 		# print("outgru", out_gru.size())
	# 		lp , idx = torch.topk(out_gru,beam_width ,dim=0,largest=True,sorted=True)
	# 		lp1 = (sc*lp)
	# 		idx = idx.tolist()
	# 		for i in range(beam_width):
	# 			if idx[i] == 1:
	# 				endval.put((lp1[i], lok+[[idx[i],lp[i]]]))
	# 			else:
	# 				nodes.put((lp1[i], lok+[[idx[i], lp[i]]]))
	# 	if endval.empty():
	# 		ll = nodes.get()
	# 		# print("ll", ll)
	# 		endval.put(ll)
	# 	ll = endval.get()[1]
	# 	# print("ll2", ll)
	# 	return ll
	
	def forward(self, word_vec):
		bs = word_vec.size(0)
		start_tok = self.emb(torch.LongTensor([[0] for _ in range(bs)]).to(device)).view(bs, -1)
		# h = self.rnn.init_hidden(bs)
		h = torch.zeros(bs, self.rnn_hidden*self.rnn_layers).view(self.rnn_layers, bs, -1)
		c = torch.zeros(bs, self.rnn_hidden*self.rnn_layers).view(self.rnn_layers, bs, -1)
		resp, resi = [], []
		for i in range(14):
			# print("start_tok size = ", start_tok.size())
			# print("word_vec size = ", word_vec.size())
			xt = torch.cat((start_tok, word_vec), dim=1).view(bs, 1, -1)
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

	# def forward3(self, word_vec):
	# 	bs = word_vec.size(0)
	# 	start_tok = self.emb(torch.LongTensor([[0] for _ in range(bs)]).to(device)).view(bs, -1)
	# 	# h = self.rnn.init_hidden(bs)
	# 	h = torch.zeros(bs, self.rnn_hidden).view(bs, self.rnn_layers, -1)
	# 	c = torch.zeros(bs, self.rnn_hidden).view(bs, self.rnn_layers, -1)
	# 	resp, resi = [], []
	# 	for i in range(10):
	# 		xt = torch.cat((start_tok, word_vec), dim=1).view(bs, 1, -1)
	# 		h = self.rnn.forward(xt, h)
	# 		# h = self.rnn(xt)
	# 		start_tok = self.lin(h).view(bs, -1)
	# 		start_tok = self.softmax(start_tok)
	# 		topk_p, topk_i = torch.topk(start_tok, 1, dim=1, largest=True)
	# 		resp.append(topk_p)
	# 		resi.append(topk_i)
	# 		start_tok = self.emb(topk_i).view(bs, -1)
	# 	return resp, resi

def make_exp_path(lr, lossParam, word_lim):
	global exp_tb_path, writer
	liss = ['lr', str(lr), 'lossParam', str(lossParam), 'wl', str(word_lim)]
	exp_tb_path = exp_tb_path_base + '_'.join(liss)
	print("exp_tb_path", exp_tb_path)
	os.system('mkdir -p '+exp_tb_path)
	os.system('mkdir -p weights')
	writer = SummaryWriter(exp_tb_path)

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

def grad_norm(model):
	total_norm = 0.0
	for p in model.parameters():
		param_norm = p.grad.data.norm(2)
		total_norm += param_norm.item() ** 2
	total_norm = total_norm ** (1. / 2)
	return total_norm



def train2():
	word_to_vec_map = np.load(w2vc_path).item()
	num_words = len(word_to_vec_map)
	print("num_words = ", num_words)
	word_lim = num_words
	lossParam = 1
	lr = 0.005 ##
	batch_size = 128 ##
	make_exp_path(lr, lossParam, word_lim)
	words = list(word_to_vec_map.keys())
	model = Model(rnn_layers=2)
	optimizer = optim.Adam(model.parameters(), lr=lr, amsgrad=True)
	itr = 0
	test_word = 'apple'
	test_embedd = torch.tensor(
		word_to_vec_map[test_word]#.astype(dtype=np.float64)

		).view(1, 1, -1).to(device)
	if os.path.isfile(save_path):
		ckpt = torch.load(save_path)
		model.load_state_dict(ckpt['model_params'])
		optimizer.load_state_dict(ckpt['optim_params'])
		itr, best_train_err = ckpt['itr'], ckpt['loss']
		del ckpt
		gc.collect()
	best_train_err = None
	itr_limit = int(1e10)
	iterr = 0
	while(iterr < itr_limit):
		optimizer.zero_grad()
		loss = torch.tensor(0.0).view(1,1)
		tot_len = 0
		for bi in range(batch_size):
			itr += 1
			# sel = 'apple'
			while(True):
				sel = random.choice(words)
				if valid_word(sel):
					break
			embedd = torch.tensor(
				word_to_vec_map[sel]#.astype(dtype=np.float64)
				).view(1, -1).to(device)
			sel = sel + "\0"
			bad = 0
			resp, resi = model.forward(embedd)
			gt_int = [get_c2i(s) for s in sel]
			len_max = len(gt_int) if len(gt_int) > len(resi) else len(resp)
			for j in range(len_max):
				if j < len(gt_int):
					if j < len(resp):
						tot_len += 1
						if (gt_int[j] == resi[j]):
							# print("pred_int[j]", pred_int[j])
							if torch.isnan(torch.log(resp[j])) or torch.isinf(torch.log(resp[j])):
								print("ISNAN HERRE")
								bad = 1
								break
							else:
								loss -= torch.log(resp[j])
						else:
							if torch.isnan(torch.log(torch.tensor(1.0).view(-1,1) - resp[j])) or torch.isinf(torch.log(torch.tensor(1.0).view(-1,1) - resp[j])):
								print("ISNAN 2")
								bad = 1
								break
							else:
								loss -= torch.log(torch.tensor(1.0).view(-1,1) - resp[j])
						if (resi[j] == 1):
							break
					else:
						loss += lossParam
				else:
					loss += lossParam
			# print("[{}/{}]")
		if tot_len == 0 :
			print("SKIPPER")
			iterr += 1
			continue

		if bad == 1:
			continue
		loss /= (tot_len+1e-3)
		# print(sel)

		try:
			print("pred_val:|{}|\t;gt:\t|{}|".format(print_word(resi), sel))
		except:
			print("Not printing due to unicode issue")
		# print("pred_val", print_word(resi), "\t;gt:\t", sel)

		loss.backward()

		nn.utils.clip_grad_norm_(model.parameters(), 0.25)
		
		print("grad_norm", grad_norm(model))
		nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.8)
		
		optimizer.step()
		# optimizer.zero_grad()

		loss_item = loss.item()
		writer.add_scalar('loss', loss_item, itr)
		print("loss: {:.5f} iter {}".format(loss_item, itr))
		if best_train_err is None or loss_item < best_train_err:
			best_train_err = loss_item
			print("Saving Weights with loss {}...".format(best_train_err))
			torch.save({
				'itr': itr+1,
				'model_params': model.state_dict(),
				'optim_params': optimizer.state_dict(),
				'loss': best_train_err
			}, save_path)
			print("Model saving Done")
		if iterr%100 == 0:
			# best_train_err = loss_item
			print("Saving Weights with loss {}...".format(best_train_err))
			torch.save({
				'itr': itr+1,
				'model_params': model.state_dict(),
				'optim_params': optimizer.state_dict(),
				'loss': loss_item
			}, save_path2)
			print("Model saving Done")
		iterr += 1
		# with torch.no_grad():


# def train():
# 	word_lim = 1000
# 	lossParam = 1
# 	lr = 0.001
# 	batch_size = 64
# 	make_exp_path(lr, lossParam, word_lim)
# 	num_words = len(word_to_vec_map)
# 	words = list(word_to_vec_map.keys())
# 	model = Model()
# 	optimizer = optim.Adam(model.parameters(), lr=lr, amsgrad=True)
# 	itr = 0
# 	test_word = 'apple'
# 	test_embedd = torch.tensor(embedding_mat[word_to_vec_map[test_word], :]).view(1, 1, -1)
# 	if os.path.isfile(save_path):
# 		ckpt = torch.load(save_path)
# 		model.load_state_dict(ckpt['model_params'])
# 		optimizer.load_state_dict(ckpt['optim_params'])
# 		itr, best_train_err = ckpt['itr'], ckpt['loss']
# 		del ckpt
# 		gc.collect()
# 	best_train_err = None
# 	for iterr in range(word_lim):
# 		optimizer.zero_grad()
# 		loss = torch.tensor(0.0)
# 		for bi in range(batch_size):
# 			itr += 1
# 			# sel = 'apple'
# 			while(True):
# 				sel = random.choice(words)
# 				if valid_word(sel):
# 					break			
# 			embedd = torch.tensor(embedding_mat[word_to_vec_map[sel], :]).view(1, 1, -1)
# 			sel = "\b" + sel + "\0"
# 			pred_int = model.forward(embedd, sel, get_c2i, tfratio=0.5, beam_width=5)
# 			# pred_val = ''.join([chr(pred[0]) for pred in pred_int if (chr(pred[0]) != 1 and chr(pred[0]) != 0)])
# 			print("pred_val", print_word(pred_int), "\t;gt:\t", sel)
# 			gt_int = [get_c2i(s) for s in sel]
# 			len_max = len(gt_int) if len(gt_int) > len(pred_int) else len(pred_int)
# 			for j in range(len_max):
# 				if j < len(gt_int):
# 					if j < len(pred_int):
# 						if (gt_int[j] == pred_int[j][0]):
# 							# print("pred_int[j]", pred_int[j])
# 							loss -= torch.log(pred_int[j][1])
# 						else:
# 							loss -= torch.log( 1.0 - pred_int[j][1])
# 					else : 
# 						loss += lossParam
# 				else : 
# 					loss += lossParam
		
# 		loss.backward()
# 		optimizer.step()
# 		loss_item = loss.item()
# 		writer.add_scalar('loss', loss_item, itr)
# 		print("loss: {:.5f} iter {}".format(loss_item, itr))
# 		if best_train_err is None or loss_item < best_train_err:
# 			best_train_err = loss_item
# 			torch.save({
# 				'itr':itr+1,
# 				'model_params': model.state_dict(),
# 				'optim_params': optimizer.state_dict(),
# 				'loss': best_train_err
# 			}, save_path)
		# with torch.no_grad():
			

train2()
