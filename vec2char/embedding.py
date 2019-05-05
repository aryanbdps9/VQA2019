import numpy as np
import os
import gc
emb_dim = 300
GloveEmbeddings = {}
embedding_mat = None
word_to_index_map = {}
juice_path = "data/"
glove_path = os.path.join(juice_path, "glove", "glove.6B.300d.txt")
embedding_path = os.path.join(juice_path, "glove", "embfile.npy")
w2im_path = os.path.join(juice_path, "glove", "w2im.npy")

def get_c2i(char):
	if (char == '\b'):
		return 0
	if (char == '\0'):
		return 1
	return 2 + ord(char)
	# if ord(char) > 64 and ord(char) < 91:
	# 	return ord(char) - 63
	# if ord(char) > 96 and ord(char) < 123:
	# 	return ord(char) - 95
	# if ord(char) > 47 and ord(char) < 58:
	# 	return ord(char) - 20
	# else:
	# 	print("gadbad hai", char)
	# 	return 4

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

def main():
	global embedding_mat, word_to_index_map
	if (os.path.exists(embedding_path)):
		# embedding_mat = np.load(embedding_path)
		# word_to_index_map = np.load(w2im_path).item()
		print("loaded files from disk")
		return
	loadEmbeddings(glove_path)
	pop_embmat_w2imap()
	print("init_embeddings done. saving files to disk")
	np.save(embedding_path, embedding_mat)
	np.save(w2im_path, word_to_index_map)
	return
main()
