import sys
sys.path.append("../utilities/")
reload(sys)
sys.setdefaultencoding('utf-8')
import os, json, gzip, operator, string, pickle, time
import numpy as np
import torch
import preprocessor as p
from torch.autograd import Variable
from multiprocessing import Process
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from utilities import get_timeList, extract_tweet_info


def collapse_wordList(valid_wordList):
    current_idx = None
    current_chunk = None
    current_importance = None
    chunkList = []
    for word in valid_wordList:
        if current_idx == None:
            current_idx = word[0]
            current_chunk = word[1]
            current_importance = word[2]
        else:
            if word[0] == current_idx + 1:
                current_idx = word[0]
                current_chunk += "_" + word[1]
                current_importance += word[2]
            else:
                chunkList.append([current_chunk, current_importance])
                current_idx = word[0]
                current_chunk = word[1]
                current_importance = word[2]
    chunkList.append([current_chunk, current_importance])
    return chunkList



def process_sentence(sent, infersent, tokenize=True):
    sent = sent.split() if not tokenize else word_tokenize(sent)
    sent = [['<s>'] + [word for word in sent if word in infersent.word_vec] + ['</s>']]
    if ' '.join(sent[0]) == '<s> </s>':
        import warnings
        warnings.warn('No words in "{0}" have glove vectors. Replacing by "<s> </s>"..'.format(sent))

    batch = Variable(infersent.get_batch(sent), volatile=True)

    # use gpu
    batch = batch.cuda()

    output = infersent.enc_lstm(batch)[0]
    output, idxs = torch.max(output, 0)
    # output, idxs = output.squeeze(), idxs.squeeze()
    idxs = idxs.data.cpu().numpy()
    argmaxs = [np.sum((idxs == k)) for k in range(len(sent[0]))]

    wordList = sent[0]
    importanceList = [float(n)/np.sum(argmaxs) for n in argmaxs] # importance percentage

    valid_wordList = []
    if len(importanceList) > 0:
        avg_importance = float(sum(importanceList)) / float(len(importanceList))
        for i in range(0, len(wordList)):
            #if importanceList[i] > avg_importance and wordList[i].lower() not in exclude and (len(wordList[i]) > 1 or wordList[i].isdigit()):
            #    valid_wordList.append([i, wordList[i].lower(), importanceList[i]])
            if importanceList[i] > avg_importance and wordList[i].lower() not in exclude and (len(wordList[i]) > 1 or wordList[i].isdigit()):
                valid_wordList.append([i, wordList[i].lower(), importanceList[i]])
        #return collapse_wordList(valid_wordList)
        return valid_wordList
    else:
        return []

    
exclude = set(string.punctuation)
exclude = exclude | set(["<s>", "</s>", "``", "''"])
exclude = exclude | set(["'m", "'s", "'re", "am", "is", "are", "'ve", "im", "n't", "yr", "damn", "shit", "fuck", "fucking", "oh", "um", "rt", "amp", "lmao"])
stop = set(stopwords.words('english'))
exclude = exclude | stop

def round_float(f):
    return "%0.4f" % f

def round_vector(v):
    new_v = []
    for n in v:
        new_v.append(float(round_float(n)))
    return new_v

def process_files(folder, i, valid_date, hourList, minuteList):
    print i, valid_date

    #sentence = "Even if you are far from the devastation of Hurricane Harvey, there are ways to contribute."
    #process_sentence(sentence, infersent)

    # use cpu
    #infersent = torch.load('../SNLI_encoder/encoder/infersent.allnli.pickle', map_location=lambda storage, loc: storage)
    #infersent.use_cuda = False
    # use gpu
    infersent = torch.load('../SNLI_encoder/encoder/infersent.allnli.pickle')
    infersent.set_glove_path("../../tools/glove.840B/glove.840B.300d.txt")
    infersent.build_vocab_k_words(K=100000)

    p.set_options(p.OPT.URL, p.OPT.EMOJI, p.OPT.MENTION, p.OPT.HASHTAG)

    output = open(valid_date + "_tweetList.txt", "w")
    tweet_id2embedding = {}
    for hour in hourList:
        #print "    ", valid_date, hour
        for minute in minuteList:
            if valid_date + "_" + str(hour) + "_" + str(minute) in ["2017_09_12_15_30", "2017_09_25_16_40"]:
                continue
            file = folder + "20170822-20170930_72kqh0j1v5_" + valid_date + "_" + str(hour) + "_" + str(minute) + "_activities.json.gz"
            if not os.path.exists(file):
                continue
            
            input_file = gzip.open(file, "r")
            for line in input_file:
                tweet = json.loads(line)
                new_tweet = extract_tweet_info(tweet)

                if "body" in new_tweet:
                    doc = new_tweet["body"]
                    cleaned_doc = doc.replace(u"\u2026", "").replace(u"RT ", "")
                    cleaned_doc = p.clean(cleaned_doc)
                    cleaned_doc = cleaned_doc.lower()
                    if len(cleaned_doc.split()) >= 3:
                        
                        chunkList = process_sentence(cleaned_doc, infersent)
                        new_tweet["chunkList"] = chunkList
                        #embeddings = infersent.encode([cleaned_doc], tokenize=True)
                        #new_tweet["embedding"] = round_vector(embeddings[0].tolist())
                        output.write(str(new_tweet) + "\n")
                        
                        embeddings = infersent.encode([cleaned_doc], tokenize=True)
                        words = tweet["link"].split("/")
                        tweet_id = words[3] + "_" + words[5]
                        
                        #tweet_id2embedding[tweet_id] = embeddings[0]
                        #tweetList.append(new_tweet)
            input_file.close()
    output.close()
    



if __name__ == "__main__":
    
    #folder = "/data/GNIPHarveyTweets/"
    folder = "../data/"

    dateList, hourList, minuteList = get_timeList()
    dateList = ["2017_08_25", "2017_08_26", "2017_08_27", "2017_08_28", "2017_08_29", "2017_09_02"]

    for j in range(0, 4):
        new_dateList = dateList[j*3:(j+1)*3]

        processV = []
        for i in range(0, len(new_dateList)):
            processV.append(Process(target = process_files, args = (folder, i, new_dateList[i], hourList, minuteList,)))
        for i in range(0, len(new_dateList)):
            processV[i].start()
        for i in range(0, len(new_dateList)):
            processV[i].join()
