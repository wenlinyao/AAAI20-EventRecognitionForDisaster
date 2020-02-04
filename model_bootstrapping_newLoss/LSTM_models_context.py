"""
This file defines the neural network structure
"""
import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np 

class BasicRNN(nn.Module):
    """
    Bi-LSTM module
    """
    def __init__(self, args, vocab_size, pretrained):
        """
        Construct a Bi-LSTM based on arguments
        """
        super(BasicRNN, self).__init__()
        self.args = args
        self.vocab_size = vocab_size

        self.drop = nn.Dropout(self.args.dropout)
        self.encoder = nn.Embedding(self.vocab_size, self.args.embedding_size)
        
        #self.rnn = nn.LSTM(input_size = self.args.embedding_size, hidden_size = self.args.rnn_size, num_layers = self.args.rnn_layers, dropout = self.args.dropout)
        #self.decoder = nn.Linear(self.args.rnn_size, self.args.class_num)

        # Bi-LSTM
        self.post_rnn = nn.LSTM(input_size = self.args.embedding_size, hidden_size = self.args.rnn_size, num_layers = self.args.rnn_layers, bidirectional = True, dropout = self.args.dropout)
        self.context_rnn = nn.LSTM(input_size = self.args.embedding_size, hidden_size = self.args.rnn_size, num_layers = self.args.rnn_layers, bidirectional = True, dropout = self.args.dropout)
        self.reply_rnn = nn.LSTM(input_size = self.args.embedding_size, hidden_size = self.args.rnn_size, num_layers = self.args.rnn_layers, bidirectional = True, dropout = self.args.dropout)
        
        self.decoder = nn.Linear(self.args.rnn_size * 2 * 3, self.args.class_num) # Other class

        self.softmax = nn.Softmax(dim=1)

        """
        # universal weight for context
        self.alpha_raw = nn.Parameter(torch.Tensor(1))
        # universal weight for reply
        self.beta_raw = nn.Parameter(torch.Tensor(1))
        nn.init.uniform_(self.alpha_raw, -0.01, 0.01)
        nn.init.uniform_(self.beta_raw, -0.01, 0.01)
        """

        self.init_weights(pretrained = pretrained)
        print "Initialized LSTM model"
    """
    def alpha(self):
        return torch.sigmoid(self.alpha_raw) #* 2
    def beta(self):
        return torch.sigmoid(self.beta_raw) #* 2
    """

    def sentence_encoder(self, input_data, hidden, flag):
        emb = self.drop(self.encoder(input_data))
        #emb = self.encoder(input_data)

        if flag == "post":
            output, hidden = self.post_rnn(emb, hidden)
        elif flag == "context":
            output, hidden = self.context_rnn(emb, hidden)
            #output, hidden = self.post_rnn(emb, hidden)
        else:
            output, hidden = self.reply_rnn(emb, hidden)

        output = self.drop(output)

        if (self.args.aggregation == "mean"):
            # compress several rows to one
            output = torch.mean(output, 0)
        elif (self.args.aggregation == "last"):
            last_idx = Variable(torch.LongTensor([output.size()[0] - 1]))
            if self.args.cuda == True:
               last_idx = last_idx.cuda()
            output = torch.index_select(output, 0, last_idx)
        elif (self.args.aggregation == "max"):
            output = torch.max(output, 0)[0]

        output = torch.squeeze(output, 0)
        return output


    def forward(self, sentences, context_sentencesList, context_wList, reply_sentencesList, reply_wList, hidden):
        """
        Define how the neural network maps input to output classes
        """
        
        sent_emb = self.sentence_encoder(sentences, hidden, "post")

        context_embList = []
        for i in range(0, len(context_wList)):
            context_embList.append(self.sentence_encoder(context_sentencesList[i], hidden, "context") * context_wList[i])

        context_emb_sum = context_embList[0]
        for context_emb in context_embList[1:]:
            context_emb_sum += context_emb

        reply_embList = []
        for i in range(0, len(reply_wList)):
            reply_embList.append(self.sentence_encoder(reply_sentencesList[i], hidden, "reply") * reply_wList[i])
        
        reply_emb_sum = reply_embList[0]
        for reply_emb in reply_embList[1:]:
            reply_emb_sum += reply_emb

        #output = sent_emb
        #output = context_emb_sum
        #output = reply_emb_sum
        #output = torch.cat((context_emb_sum, sent_emb), 1)
        #output = torch.cat((reply_emb_sum, sent_emb), 1)
        output = torch.cat((context_emb_sum, reply_emb_sum, sent_emb), 1)
        
        #output = torch.cat((context_emb_sum * self.alpha(), reply_emb_sum * self.beta(), sent_emb), 1)
        #output = torch.cat((context_emb_sum, reply_emb_sum), 1)
        
        output = self.drop(output)
        
        decoded = self.decoder(output)

        # batch_size * class_num
        #prob = self.softmax(decoded)
        prob = decoded

        return prob, hidden

    def init_weights(self, pretrained):
        """
        Initialize weights using pretrained word embedding (e.g., GloVe)
        """
        initrange = 0.1
        print("Setting pretrained embeddings")
        pretrained = pretrained.astype(np.float32)
        pretrained = torch.from_numpy(pretrained)
        if self.args.cuda == True:
            pretrained = pretrained.cuda()
        self.encoder.weight.data.copy_(pretrained)
        self.encoder.weight.requires_grad = self.args.trainable
        self.decoder.bias.data.fill_(0)
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def init_hidden(self, batch_size):
        """
        Initialize hidden states for LSTM
        """
        #h0 = Variable(torch.zeros(self.args.rnn_layers, batch_size, self.args.rnn_size))
        #c0 = Variable(torch.zeros(self.args.rnn_layers, batch_size, self.args.rnn_size))

        # Bi-LSTM
        h0 = Variable(torch.zeros(self.args.rnn_layers * 2, batch_size, self.args.rnn_size))
        c0 = Variable(torch.zeros(self.args.rnn_layers * 2, batch_size, self.args.rnn_size))
        
        if self.args.cuda == True:
            return (h0.cuda(), c0.cuda())
        else:
            return (h0, c0)
