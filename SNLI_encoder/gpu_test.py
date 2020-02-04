import torch
from torch.autograd import Variable
import numpy as np
import time
from nltk.tokenize import word_tokenize



def process_sentence(sent, infersent, tokenize=True):
    sent = sent.split() if not tokenize else word_tokenize(sent)
    sent = [['<s>'] + [word for word in sent if word in infersent.word_vec] + ['</s>']]
    if ' '.join(sent[0]) == '<s> </s>':
        import warnings
        warnings.warn('No words in "{0}" have glove vectors. Replacing by "<s> </s>"..'.format(sent))

    batch = Variable(infersent.get_batch(sent), volatile=True)

    batch = batch.cuda()
    output = infersent.enc_lstm(batch)[0]
    output, idxs = torch.max(output, 0)
    # output, idxs = output.squeeze(), idxs.squeeze()
    idxs = idxs.data.cpu().numpy()
    argmaxs = [np.sum((idxs == k)) for k in range(len(sent[0]))]

    wordList = sent[0]
    importanceList = [float(n)/np.sum(argmaxs) for n in argmaxs] # importance percentage

    return importanceList

if __name__ == "__main__":
    infersent = torch.load('encoder/infersent.allnli.pickle')
    #infersent = torch.load('encoder/infersent.allnli.pickle', map_location=lambda storage, loc: storage)

    infersent.set_glove_path("../../tools/glove.840B/glove.840B.300d.txt")
    infersent.build_vocab_k_words(K=100000)

    sentence = "Even if you are far from the devastation of Hurricane Harvey, there are ways to contribute."

    start = time.time()
    for i in range(0, 1000):
        process_sentence(sentence, infersent)
    end = time.time()
    print (end - start)