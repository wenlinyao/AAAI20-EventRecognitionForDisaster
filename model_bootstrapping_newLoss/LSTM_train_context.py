"""
This file defines:
1. how to split data into train and development
2. make all training data into batches
3. train the neural network
"""

import LSTM_models_context
from LSTM_models_context import BasicRNN
import cPickle as pickle
import random, time, math
import numpy as np
import torch
from tqdm import tqdm
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
from keras.preprocessing import sequence
import copy

# https://github.com/vanzytay/pytorch_sentiment_rnn

def tensor_to_numpy(x):
    ''' Need to cast before calling numpy()
    '''
    #return (Variable(x).data).cpu().numpy()
    return x.data.type(torch.DoubleTensor).numpy()


class Experiment:
    """
    The basic experiment class
    """
    def __init__(self, dst_folder, args):
        """
        :param dst_folder: output folder to store data
        :param args: arguments to control the experiment setting (GPU, toy_mode, etc.)
        :return: returns nothing
        """
        self.dst_folder = dst_folder
        self.args = args

        torch.manual_seed(args.seed)
        np.random.seed(args.seed)

        with open(self.dst_folder + "env.pkl",'r') as f:
            self.env = pickle.load(f)

        self.drop_indices = self.env['drop_indices']
        self.original_train_set = self.env['train']
        self.train_set = copy.deepcopy(self.original_train_set)
        self.dev_set = self.env['dev']
        self.test_set = self.env['test']

        if(self.args.toy == True):
            print("Using toy mode...")
            random.shuffle(self.train_set)
            self.train_set = self.train_set[:1000]
            self.dev_set = self.dev_set[:200]

            random.shuffle(self.test_set)
            self.test_set = self.test_set[:200]

        if(self.args.dev==0):
            self.train_set = self.train_set + self.dev_set

        classes_freq = [1 for i in range(0, self.args.class_num)]
        for instance in self.train_set:
            for i in range(0, len(instance["class"])):
                if instance["class"][i] == 1:
                    classes_freq[i] += 1
        classes_freq_sum = sum(classes_freq)

        #classes_weight = [math.log(float(classes_freq_sum)/float(freq)) for freq in classes_freq]

        classes_weight = [float(classes_freq_sum)/float(freq) for freq in classes_freq]

        #classes_weight = [1.0 for freq in classes_freq]

        self.classes_weight = torch.from_numpy(np.array(classes_weight, dtype='float32'))
        print "classes_freq:", classes_freq
        print "classes_weight:", classes_weight

        if torch.cuda.is_available():
            if self.args.cuda == False:
                print("WARNING: You have a CUDA device, so you should probably run with --cuda")
            else:
                print ("There are {} CUDA devices".format(self.args.gpu))
                if(self.args.gpu > 0):
                    print("Setting torch GPU to {}".format(self.args.gpu))
                    torch.cuda.set_device(self.args.gpu)
                    print("Using device:{} ".format(torch.cuda.current_device()))
                torch.cuda.manual_seed(self.args.seed)

        self.mdl = BasicRNN(self.args, len(self.env["word_index"]), pretrained = self.env["glove"])
        
        if self.args.trained_model == None: # no pretrained model
            print "creating model..."

        else: # use pretrained model
            print "loading trained model..."
            self.mdl.load_state_dict(torch.load(self.dst_folder + self.args.trained_model))


        if self.args.cuda == True:
            self.mdl.cuda()
            self.classes_weight = self.classes_weight.cuda()


    def select_optimizer(self):
        """
        Select different optimizers for the neural network
        """
        parameters = filter(lambda p: p.requires_grad, self.mdl.parameters())
        
        if(self.args.opt=='Adam'):
            self.optimizer =  optim.Adam(parameters, lr=self.args.learn_rate)
        elif(self.args.opt=='RMS'):
            self.optimizer =  optim.RMSprop(parameters, lr=self.args.learn_rate)
        elif(self.args.opt=='SGD'):
            self.optimizer =  optim.SGD(parameters, lr=self.args.learn_rate)
        elif(self.args.opt=='Adagrad'):
            self.optimizer =  optim.Adagrad(parameters, lr=self.args.learn_rate)
        elif(self.args.opt=='Adadelta'):
            self.optimizer =  optim.Adadelta(parameters, lr=self.args.learn_rate)


    def evaluate(self, x):
        '''
        Evaluates normal RNN model and calculates confusion matrix
        '''
        self.mdl.eval()

        if len(x) % self.args.batch_size == 0:
            num_batches = int(len(x) / self.args.batch_size)
        else:
            num_batches = int(len(x) / self.args.batch_size) + 1

        all_probs = []
        all_preds = []
        all_targets = []
        for instance in x:
            all_targets.append(instance["class"])

        for i in range(num_batches):
            sentences, context_sentencesList, context_wList, reply_sentencesList, reply_wList, targets, actual_batch_size = self.make_batch(x, i, evaluation=True)
            hidden = self.mdl.init_hidden(actual_batch_size)
            output, hidden = self.mdl(sentences, context_sentencesList, context_wList, reply_sentencesList, reply_wList, hidden)
            all_probs += tensor_to_numpy(output).tolist()

        for probs in all_probs:
            pred = []
            for i, p in enumerate(probs):
                if p > 0:
                    pred.append(i)
            
            all_preds.append(pred)
        
        print "len(all_targets):", len(all_targets), "len(all_preds):", len(all_preds)
        confusion_matrix = {}
        matches = 0
        for i in range(len(all_targets)):
            if all_targets[i] == all_preds[i]:
                matches += 1
            string = str(all_targets[i]) + " --> " + str(all_preds[i])
            if string in confusion_matrix:
                confusion_matrix[string] += 1
            else:
                confusion_matrix[string] = 1
        acc = float(matches) / float(len(all_targets))
        
        print "confusion_matrix[target --> pred]:", confusion_matrix
        return all_probs, all_preds, acc
        

    def pad_to_batch_max(self, x, max_len):
        """
        Pad the input sentences to the same maximum length in order to support mini-batch
        """
        #lengths = [len(y) for y in x]
        #max_len = np.max(lengths)
        padded_tokens = sequence.pad_sequences(x, maxlen=max_len)
        #print "padded_tokens:", padded_tokens
        return torch.LongTensor(padded_tokens.tolist()).transpose(0,1)

    def make_batch(self, x, i, evaluation=False):
        ''' 
        :param x: input sentences
        :param i: select the ith batch (-1 to take all)
        :return: sentences, targets, actual_batch_size
        '''
        if(i>=0):
            batch = x[int(i * self.args.batch_size):int((i + 1) * self.args.batch_size)]
        else:
            batch = x
        
        if(len(batch)==0):
            return None, None, self.args.batch_size

        sentences = self.pad_to_batch_max([x['tokenized_txt'] for x in batch], self.args.maxlen)
        context_sentencesList = []
        context_wList = []
        if self.args.context_size == 0:
            # if context_size == 0, put a padding context sentence
            context_len = self.args.context_size + 1
        else:
            context_len = self.args.context_size

        for i in range(0, context_len):
            context_sentences = []
            context_w = []
            for x in batch:
                context_sentences.append(x["context_dataList"][i][0])
                context_w.append(x["context_dataList"][i][1])
            assert sentences.size(1) == len(context_sentences)
            context_sentencesList.append(self.pad_to_batch_max(context_sentences, self.args.maxlen))
            context_wList.append(context_w)


        reply_sentencesList = []
        reply_wList = []
        if self.args.context_size == 0:
            reply_len = self.args.context_size + 1
        else:
            reply_len = self.args.context_size

        for i in range(0, reply_len):
            reply_sentences = []
            reply_w = []
            for x in batch:
                reply_sentences.append(x["reply_dataList"][i][0])
                reply_w.append(x["reply_dataList"][i][1])
            assert sentences.size(1) == len(reply_sentences)
            reply_sentencesList.append(self.pad_to_batch_max(reply_sentences, self.args.maxlen))
            reply_wList.append(reply_w)

        #targets = torch.LongTensor(np.array([x['class'] for x in batch], dtype=np.int32).tolist())

        targets = torch.from_numpy(np.array([x['class'] for x in batch], dtype=np.float32))

        for i in range(0, context_len):
            context_wList[i] = torch.from_numpy(np.array(context_wList[i], dtype = 'float32')).view(-1, 1)

        for i in range(0, reply_len):
            reply_wList[i] = torch.from_numpy(np.array(reply_wList[i], dtype = 'float32')).view(-1, 1)

        if self.args.cuda == True:
            sentences = sentences.cuda()
            for i in range(0, context_len):
                context_sentencesList[i] = context_sentencesList[i].cuda()
            for i in range(0, context_len):
                context_wList[i] = context_wList[i].cuda()

            for i in range(0, reply_len):
                reply_sentencesList[i] = reply_sentencesList[i].cuda()
            for i in range(0, reply_len):
                reply_wList[i] = reply_wList[i].cuda()

            targets = targets.cuda()

        actual_batch_size = sentences.size(1)


        sentences = Variable(sentences, volatile=evaluation)

        for i in range(0, context_len):
            context_sentencesList[i] = Variable(context_sentencesList[i], volatile=evaluation)

        for i in range(0, reply_len):
            reply_sentencesList[i] = Variable(reply_sentencesList[i], volatile=evaluation)

        targets = Variable(targets, volatile=evaluation)

        #sentences = Variable(sentences)
        #targets = Variable(targets)

        return sentences, context_sentencesList, context_wList, reply_sentencesList, reply_wList, targets, actual_batch_size

    def train_batch(self, i):
        '''
        Trains a regular RNN model
        '''
        #print self.make_batch(self.train_set, i)
        sentences, context_sentencesList, context_wList, reply_sentencesList, reply_wList, targets, actual_batch_size = self.make_batch(self.train_set, i)

        
        if(sentences is None):
            return None
        
        hidden = self.mdl.init_hidden(actual_batch_size)

        #self.mdl.zero_grad()
        self.optimizer.zero_grad()

        output, hidden = self.mdl(sentences, context_sentencesList, context_wList, reply_sentencesList, reply_wList, hidden)
        
        #print "output:", output
        #print "targets:", targets
        loss = self.criterion(output, targets)
        
        loss.backward()

        nn.utils.clip_grad_norm_(parameters = self.mdl.parameters(), max_norm = self.args.clip)
        self.optimizer.step()

        #return loss.data[0]
        return loss.item()

    def reprocess_train_set(self):
        self.train_set = copy.deepcopy(self.original_train_set)
        for i in range(0, len(self.train_set)):
            for j in range(0, len(self.train_set[i]["tokenized_txt"])):
                if self.train_set[i]["tokenized_txt"][j] in self.drop_indices and random.uniform(0, 1) < self.args.keywords_drop_r:
                    self.train_set[i]["tokenized_txt"][j] = 0 # index of <pad>
        random.shuffle(self.train_set)


    def train(self):
        """
        This is the main train function
        """
        if self.args.trained_model != None:
            print "Use pretrained model. No training."
            return

        #self.criterion = nn.CrossEntropyLoss(weight = self.classes_weight)
        self.criterion = nn.MultiLabelSoftMarginLoss(weight = self.classes_weight)
        print(self.args)
        total_loss = 0

        if len(self.train_set) % self.args.batch_size == 0:
            num_batches = int(len(self.train_set) / self.args.batch_size)
        else:
            num_batches = int(len(self.train_set) / self.args.batch_size) + 1

        print "len(self.train_set)", len(self.train_set)
        print "num_batches:", num_batches
        self.select_optimizer()

        best_acc = 0
        for epoch in range(1, self.args.epochs+1):
            self.mdl.train()
            print "epoch: ", epoch
            t0 = time.clock()

            #random.shuffle(self.train_set)
            self.reprocess_train_set()

            print("========================================================================")
            losses = []
            actual_batch = self.args.batch_size
            for i in tqdm(range(num_batches)):
                loss = self.train_batch(i)
                if(loss is None):
                    continue    
                losses.append(loss)
            t1 = time.clock()
            print("[Epoch {}] Train Loss={} T={}s".format(epoch, np.mean(losses),t1-t0 ))
            if(epoch % self.args.eval == 0):
                print "Evaluate on dev set..."
                all_probs, all_preds, acc = self.evaluate(self.dev_set)
                print "accuracy:", acc
        
        torch.save(self.mdl.state_dict(), self.dst_folder + "model.pt")

                #if acc > best_acc:
                #    best_acc = acc
                #    torch.save(self.mdl.state_dict(), self.dst_folder + "best_acc_model.pt")
    
    def test(self):
        """
        Test the model and output the result to "LSTM_true_and_pred_value.txt"
        """
        all_probs, all_preds, acc = self.evaluate(self.test_set)
        #all_probs, all_preds = self.evaluate(self.dev_set)

        output_file = open("LSTM_true_and_pred_value.txt", "w")
        for i, instance in enumerate(self.test_set):
        #for i, instance in enumerate(self.dev_set):
            output_file.write(instance["original_text"] + "\t" + str(instance["tweet_id"]) + "\t" + str(all_preds[i]) + "\t" + str(all_probs[i]) + "\n")
        output_file.close()
        



#if __name__ == '__main__':
def LSTM_main_context(current_folder, args):
    exp = Experiment(current_folder, args)
    print("Training...")
    exp.train()

    print("Evaluate on test set...")
    exp.test()

    


