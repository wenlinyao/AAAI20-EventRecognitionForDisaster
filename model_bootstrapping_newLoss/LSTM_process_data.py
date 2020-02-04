"""
This file prepares training data (including vocabulary, word embedding lookup from GloVe) for context-independent classifier.
"""
import sys
sys.path.append("../utilities/")
reload(sys)
sys.setdefaultencoding('utf-8')
import numpy as np
from tqdm import tqdm
import nltk, os, glob, re, string, ast
import cPickle as pickle
import random
from operator import itemgetter
from string import punctuation
from multiprocessing import Process
import preprocessor as p
from utilities import tweet_filter, load_important_words, postedTime2seconds
from bootstrapping_utils import build_data, tokenize

def string_standardize(myString): # remove punctuation and return lower case
    return myString.lower().translate(None, string.punctuation)


def strip_punctuation(s):
    return s.translate(string.maketrans("",""), string.punctuation)
    # return ''.join(c for c in s if c not in punctuation)




def get_context_dataList(actor, tweet_id, actor2tweet_ids, tweet_id2tokenized_txt, tweet_id2tweet, window):
    index = None
    anchor_time = None
    tweet_tuples = actor2tweet_ids[actor]
    for i in range(0, len(tweet_tuples)):
        if tweet_tuples[i][0] == tweet_id:
            index = i
            anchor_time = tweet_tuples[i][1]
            break
    context_dataList = []
    context_textList = []
    for i in range(1, window + 1):
        j = index - i
        if j >= 0 and tweet_tuples[j][0] in tweet_id2tokenized_txt:
            context_data = [tweet_id2tokenized_txt[tweet_tuples[j][0]], float(anchor_time - tweet_tuples[j][1]) / 60.0]
            context_dataList.append(context_data)
            # add text to context_textList
            if tweet_tuples[j][0] in tweet_id2tweet:
                context_textList.append(tweet_id2tweet[tweet_tuples[j][0]]["body"])
        else:
            # padding
            context_data = [[0], 999]
            context_dataList.append(context_data)
    
    # all padding weights are 0
    weightList = []
    for context_data in context_dataList:
        
        if context_data[1] != 999:
            weightList.append(0.7 ** context_data[1])
        else:
            weightList.append(0)
        
        #weightList.append(0.7 ** context_data[1])

    for i in range(0, window):
        if sum(weightList) != 0:
            context_dataList[i][1] = weightList[i] / sum(weightList) * max(weightList)
            #context_dataList[i][1] = weightList[i]
        else:
            context_dataList[i][1] = 0

    if len(context_dataList) == 0:
        context_dataList = [[[0], 0]]

    return context_dataList, context_textList

def get_reply_dataList(tweet_id, original_tweet_id2reply_ids, tweet_id2tweet, word_index, window):

    anchor_time = postedTime2seconds(tweet_id2tweet[tweet_id]["postedTime"])
    
    reply_dataList = []
    reply_textList = []

    if tweet_id in original_tweet_id2reply_ids:
        reply_tweet_ids = original_tweet_id2reply_ids[tweet_id]
        #print "len(reply_tweet_ids):", len(reply_tweet_ids)
        
        for reply_tweet_id in reply_tweet_ids:
            if len(reply_dataList) >= window:
                break
            if reply_tweet_id in tweet_id2tweet:
                tweet = tweet_id2tweet[reply_tweet_id]
                
                # add text to reply_textList
                reply_textList.append(tweet["body"])

                tokenized_text = tokenize(tweet, "lemma")

                if len(tokenized_text) <= 1:
                    continue

                tokenized_txt = []
                for x in tokenized_text:
                    if x in word_index:
                        tokenized_txt += [word_index[x]]
                    else:
                        tokenized_txt += [word_index['<unk>']]

                current_time = postedTime2seconds(tweet["postedTime"])
                #print anchor_time_, tweet["postedTime"]
                assert current_time >= anchor_time
                reply_data = [tokenized_txt, float(current_time - anchor_time) / 60.0]

                reply_dataList.append(reply_data)

    # padding
    if len(reply_dataList) < window:
        for i in range(0, window - len(reply_dataList)):
            reply_data = [[0], 999]
            reply_dataList.append(reply_data)
    
    weightList = []
    for reply_data in reply_dataList:
        
        if reply_data[1] != 999:
            weightList.append(0.99 ** reply_data[1])
        else:
            weightList.append(0)
        
        #weightList.append(0.9 ** reply_data[1])

    for i in range(0, window):
        if sum(weightList) != 0:
            reply_dataList[i][1] = weightList[i] / sum(weightList) * max(weightList)
        else:
            reply_dataList[i][1] = 0

    if len(reply_dataList) == 0:
        # only one element [[0], 0]
        reply_dataList = [[[0], 0]]

    return reply_dataList, reply_textList

def get_reply_dataList2(tweet_id, original_tweet_id2reply_ids, tweet_id2tweet, word_index, window):

    anchor_time = postedTime2seconds(tweet_id2tweet[tweet_id]["postedTime"])

    tweet_tokenized_text = tokenize(tweet_id2tweet[tweet_id], "lemma")
    
    reply_dataList = []
    reply_dataList_tmp = []

    reply_textList = []

    if tweet_id in original_tweet_id2reply_ids:
        reply_tweet_ids = original_tweet_id2reply_ids[tweet_id]
        #print "len(reply_tweet_ids):", len(reply_tweet_ids)
        
        for reply_tweet_id in reply_tweet_ids:
            if len(reply_dataList) >= window:
                break
            if reply_tweet_id in tweet_id2tweet:
                tweet = tweet_id2tweet[reply_tweet_id]
                
                # add text to reply_textList
                reply_textList.append(tweet["body"])

                tokenized_text = tokenize(tweet, "lemma")

                if len(tokenized_text) <= 1:
                    continue

                tokenized_txt = []
                for x in tokenized_text:
                    if x in word_index:
                        tokenized_txt += [word_index[x]]
                    else:
                        tokenized_txt += [word_index['<unk>']]

                current_time = postedTime2seconds(tweet["postedTime"])
                #print anchor_time_, tweet["postedTime"]
                assert current_time >= anchor_time
                reply_data = [len(set(tokenized_text) & set(tweet_tokenized_text)), tokenized_txt, float(current_time - anchor_time) / 60.0]
                reply_dataList_tmp.append(reply_data)
                #reply_dataList.append(reply_data)
    sorted_reply_dataList_tmp = sorted(reply_dataList_tmp, key = itemgetter(0), reverse = True)
    for item in sorted_reply_dataList_tmp[:window]:
        reply_dataList.append([item[1], item[2]])

    # padding
    if len(reply_dataList) < window:
        for i in range(0, window - len(reply_dataList)):
            reply_data = [[0], 999]
            reply_dataList.append(reply_data)
    
    weightList = []
    for reply_data in reply_dataList:
        
        if reply_data[1] != 999:
            weightList.append(0.99 ** reply_data[1])
        else:
            weightList.append(0)
        
        #weightList.append(0.9 ** reply_data[1])

    for i in range(0, window):
        if sum(weightList) != 0:
            reply_dataList[i][1] = weightList[i] / sum(weightList) * max(weightList)
        else:
            reply_dataList[i][1] = 0

    if len(reply_dataList) == 0:
        # only one element [[0], 0]
        reply_dataList = [[[0], 0]]

    return reply_dataList, reply_textList


def create_database_context(tweet_id2tweet, labelList, tweet_id2categories, actor2tweet_ids, original_tweet_id2reply_ids,
        train_tweet_ids, test_tweet_ids, category2important_words, w2v_file, args):
    print "loading training and testing data..."

    important_words = set()
    for category in category2important_words:
        important_words = important_words | set(category2important_words[category])

    train_sentences, test_sentences, vocab = build_data(tweet_id2tweet, tweet_id2categories, train_tweet_ids, test_tweet_ids, important_words, args.neg_sample_r, args.word_flag)

    # Building vocab indices
    index_word = {index+2:word for index,word in enumerate(vocab)}
    word_index = {word:index+2 for index,word in enumerate(vocab)}
    index_word[0], index_word[1] = '<pad>','<unk>'
    word_index['<pad>'], word_index['<unk>'] = 0,1

    training_set, testing_set = [], []

    tweet_id2tokenized_txt = {}

    for sent in train_sentences + test_sentences:
        tokenized_txt = [word_index[x] for x in sent["tokenized_text"]]
        tweet_id2tokenized_txt[sent["tweet_id"]] = tokenized_txt

    # output some samples
    output = open("tweet_context_reply_examples.txt", "w")
    valid_count = 0
    invalid_count = 0

    for sent in train_sentences:
        tokenized_txt = [word_index[x] for x in sent["tokenized_text"]]
        class_vec = [0 for i in range(0, len(labelList))]
        for l in sent["label"]:
            class_vec[labelList.index(l)] = 1
        
        tmp = {"original_text": sent["original_text"], "tokenized_txt": tokenized_txt, "tweet_id": sent["tweet_id"], 
            "actor": sent["actor"], "class": class_vec}

        context_dataList, context_textList = get_context_dataList(sent["actor"], sent["tweet_id"], actor2tweet_ids, tweet_id2tokenized_txt, tweet_id2tweet, args.context_size)
        #reply_dataList, reply_textList = get_reply_dataList(sent["tweet_id"], original_tweet_id2reply_ids, tweet_id2tweet, word_index, args.context_size)
        reply_dataList, reply_textList = get_reply_dataList2(sent["tweet_id"], original_tweet_id2reply_ids, tweet_id2tweet, word_index, args.context_size)

        # output samples
        if "#Other" not in sent["label"]:
            if len(context_textList) != 0 or len(reply_textList) != 0:
                valid_count += 1
                output.write(str(valid_count) + " " + tmp["tweet_id"] + "\n")
                output.write(tmp["original_text"] + "\n")
                output.write("################ context ################\n")
                for text in context_textList:
                    output.write(text + "\n")
                output.write("################ reply ################\n")
                for text in reply_textList:
                    output.write(text + "\n")
                output.write("\n\n")
            else:
                invalid_count += 1

        tmp["context_dataList"] = context_dataList
        tmp["reply_dataList"] = reply_dataList

        """
        if sum([item[1] for item in context_dataList]) == 0 and sum([item[1] for item in reply_dataList]) == 0:
            continue
        """

        training_set.append(tmp)

    output.write("valid_count, invalid_count:" + str(valid_count) + " " + str(invalid_count))
    output.close()


    for tmp in training_set[:100]:
        print tmp["context_dataList"], tmp["reply_dataList"]

    """
    for tmp in training_set:
        if "reservoirs" in tmp["original_text"]:
            print tmp
    """

    random.shuffle(training_set)

    for sent in test_sentences:
        tokenized_txt = [word_index[x] for x in sent["tokenized_text"]]
        class_vec = [0 for i in range(0, len(labelList))]
        for l in sent["label"]:
            class_vec[labelList.index(l)] = 1

        tmp = {"original_text": sent["original_text"], "tokenized_txt": tokenized_txt, "tweet_id": sent["tweet_id"], 
            "actor": sent["actor"], "class": class_vec}

        context_dataList, _ = get_context_dataList(sent["actor"], sent["tweet_id"], actor2tweet_ids, tweet_id2tokenized_txt, tweet_id2tweet, args.context_size)
        #reply_dataList, _ = get_reply_dataList(sent["tweet_id"], original_tweet_id2reply_ids, tweet_id2tweet, word_index, args.context_size)
        reply_dataList, _ = get_reply_dataList2(sent["tweet_id"], original_tweet_id2reply_ids, tweet_id2tweet, word_index, args.context_size)

        tmp["context_dataList"] = context_dataList
        tmp["reply_dataList"] = reply_dataList

        """
        if sum([item[1] for item in context_dataList]) == 0 and sum([item[1] for item in reply_dataList]) == 0:
            continue
        """

        testing_set.append(tmp)

    print("{} unique words".format(len(vocab)))
    print("{} train sentences".format(len(training_set)))
    print("{} test sentences".format(len(testing_set)))

    print("Spliting into Dev Set")
    random.shuffle(training_set)
    dev_len = len(training_set) // 20
    dev_set = training_set[:dev_len]
    training_set = training_set[dev_len:]

    while (len(training_set) % args.batch_size) in [0, 1]:
        training_set += [training_set[-1]]
    while (len(testing_set) % args.batch_size) in [0, 1]:
        testing_set += [testing_set[-1]]
    while (len(dev_set) % args.batch_size) in [0, 1]:
        dev_set += [dev_set[-1]]

    drop_indices = set()

    for important_word in important_words:
        if important_word in word_index:
            drop_indices.add(word_index[important_word])

    env = {
    "index_word":index_word,
    "word_index":word_index,
    "drop_indices":drop_indices,
    "train":training_set,
    "dev":dev_set,
    "test":testing_set,
    }

    
    glove = {}
    glove_path = "glove_embeddings.pkl"

    with open(w2v_file) as f:
        for l in f:
            vec = l.split(' ')
            word = vec[0].lower()
            vec = vec[1:]
            #print(word)
            #print(len(vec))
            glove[word] = np.array(vec)

    print('glove size={}'.format(len(glove)))
    save = True
    
    print("Finished making glove dictionary")

    vocab_size = len(vocab)
    dimensions = 300
    matrix = np.zeros((len(word_index), dimensions))
    #print(matrix.shape)

    oov = 0 

    filtered_glove = {}

    for i in tqdm(range(2, len(word_index))):
        word = index_word[i].lower()
        if(word in glove):
            vec = glove[word]
            if(save==True):
                filtered_glove[word] = glove[word]
            # print(vec.shape)
            #matrix = np.vstack((matrix,vec))
            matrix[i] = vec
        else:
            random_init = np.random.uniform(low=-0.01,high=0.01, size=(1,dimensions))
            # print(random_init)
            #matrix = np.vstack((matrix,random_init))
            matrix[i] = random_init
            oov +=1
            # print(word)

    if(save==True):
        with open(glove_path,'w') as f:
            pickle.dump(filtered_glove, f)
        print("Saving glove dict to file")
    

    print(matrix.shape)
    print(len(word_index))
    print("oov={}".format(oov))

    print("Saving glove vectors")
    env['glove'] = matrix

    with open("env.pkl", "w") as f:
        pickle.dump(env, f)

    print "dataset created!"

