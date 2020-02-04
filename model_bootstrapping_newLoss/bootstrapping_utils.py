import sys
sys.path.append("../utilities/")
reload(sys)
sys.setdefaultencoding('utf-8')
from utilities import tweet_filter, test_tweet_filter, process_text, postedTime2seconds
import ast, random, glob, copy, pickle, math
import preprocessor as p
from operator import itemgetter
import nltk
from nltk.stem.wordnet import WordNetLemmatizer

lmtzr = WordNetLemmatizer()

def keywords_tweet_id2categories(folder, input_name2train_hourList, category2important_words):
    tweet_id2categories = {}
    for input_name in input_name2train_hourList:
        input_file = open(folder + input_name, "r")
        hourList = input_name2train_hourList[input_name]

        print "keywords classify tweets..."

        for line in input_file:
            tweet = ast.literal_eval(line)
            
            if test_tweet_filter(tweet, hourList) == True:
                continue
            
            words = tweet["link"].split("/")
            tweet_id = words[3] + "_" + words[5]

            actor = "@" + words[3]

            cleaned_tweet_words = p.clean(tweet["body"])
            cleaned_tweet_words = process_text(cleaned_tweet_words)
            cleaned_tweet_words = set(cleaned_tweet_words.split())

            for category in category2important_words:
                if len(category2important_words[category] & cleaned_tweet_words) != 0:
                    if tweet_id not in tweet_id2categories:
                        tweet_id2categories[tweet_id] = set()
                        tweet_id2categories[tweet_id].add(category)
                    else:
                        tweet_id2categories[tweet_id].add(category)
                        
        input_file.close()

    return tweet_id2categories

# use SLPA + top 10 words to find category labels
def SLPA_tweet_id2categories(tweet_id2labels_file):
    tweet_id2labels = pickle.load(open(tweet_id2labels_file, "rb"))
    #                       0                        1                      2               3            4    
    labelList = ["#Natural_environment", "#Preventative_measure", "#Help_and_rescue", "#Casualty", "#Housing", 
    #                       5                        6                         7
                "#Utilities_and_Supplies", "#Transportation", "#Flood_control_infrastructures", 
    #                      8                          9                   10
                "#Business_Work_School", "#Built-environment_hazards", "#Other"]
    tweet_id2categories = {}

    for tweet_id in tweet_id2labels:
        tweet_id2categories[tweet_id] = set()
        for label in tweet_id2labels[tweet_id]:
            tweet_id2categories[tweet_id].add(labelList[label])

    return tweet_id2categories


def prepare_data(source_folder, input_name2train_hourList, input_name2test_hourList):
    train_tweet_ids = set()
    test_tweet_ids = set()
    actor2tweet_ids = {}
    tweet_id2tweet = {}

    for input_name in input_name2train_hourList:
        train_hourList = input_name2train_hourList[input_name]

        input_file = open(source_folder + input_name, "r")

        for line in input_file:
            tweet = ast.literal_eval(line)
            if tweet_filter(tweet, train_hourList) != True:
                train_tweet_ids.add(tweet["tweet_id"])

            actor = tweet["link"].split("/")[3]
            if actor not in actor2tweet_ids:
                actor2tweet_ids[actor] = [[tweet["tweet_id"], postedTime2seconds(tweet["postedTime"])]]
            else:
                actor2tweet_ids[actor] += [[tweet["tweet_id"], postedTime2seconds(tweet["postedTime"])]]

            tweet_id2tweet[tweet["tweet_id"]] = tweet

        input_file.close()

    for input_name in input_name2test_hourList:
        test_hourList = input_name2test_hourList[input_name]

        input_file = open(source_folder + input_name, "r")

        for line in input_file:
            tweet = ast.literal_eval(line)

            if test_tweet_filter(tweet, test_hourList) != True:
                test_tweet_ids.add(tweet["tweet_id"])

            actor = tweet["link"].split("/")[3]
            if actor not in actor2tweet_ids:
                actor2tweet_ids[actor] = [[tweet["tweet_id"], postedTime2seconds(tweet["postedTime"])]]
            else:
                actor2tweet_ids[actor] += [[tweet["tweet_id"], postedTime2seconds(tweet["postedTime"])]]

            tweet_id2tweet[tweet["tweet_id"]] = tweet

        input_file.close()

    for actor in actor2tweet_ids:
        actor2tweet_ids[actor] = sorted(actor2tweet_ids[actor], key = itemgetter(1))

    return train_tweet_ids, test_tweet_ids, actor2tweet_ids, tweet_id2tweet




def get_eval_tweet_id2categories(tweet_id2categories, test_tweet_ids):
    eval_tweet_id2categories = {}
    for tweet_id in test_tweet_ids:
        if tweet_id in tweet_id2categories:
            eval_tweet_id2categories[tweet_id] = tweet_id2categories[tweet_id]
        else:
            eval_tweet_id2categories[tweet_id] = set(["#Other"])
    return eval_tweet_id2categories

def tokenize(tweet, flag):
    cleaned_tweet_words = p.clean(tweet["body"])
    cleaned_tweet_words = process_text(cleaned_tweet_words)
    cleaned_tweet_words = cleaned_tweet_words.lower().split()
    if flag == "lemma":
        cleaned_tweet_words = [lmtzr.lemmatize(word) for word in cleaned_tweet_words]
    return cleaned_tweet_words

def scan_important_words(tweet_body, important_words):
    for word in important_words:
        if word in tweet_body:
            return True
    return False

def build_data(tweet_id2tweet, tweet_id2categories, train_tweet_ids, test_tweet_ids, important_words, neg_sample_r, word_flag):
    train_sentences, test_sentences = [], []
    vocab = []

    #random.seed(111)

    multi_label_pos_count = 0
    pos_count = 0
    neg_count = 0

    for tweet_id in train_tweet_ids:
        tweet = tweet_id2tweet[tweet_id]

        cleaned_tweet_words = tokenize(tweet, word_flag)

        #cleaned_tweet_words = [chunk[1] for chunk in tweet["chunkList"]]

        s = {"original_text": ' '.join(tweet["body"].split()), "tokenized_text": cleaned_tweet_words, 
            "tweet_id": tweet["tweet_id"], "actor": tweet["link"].split("/")[3]}

        """
        cleaned_tweet_words_removed = [word for word in cleaned_tweet_words if word not in important_words]
        s = {"original_text": ' '.join(tweet["body"].split()), "tokenized_text": cleaned_tweet_words_removed, 
            "tweet_id": tweet["tweet_id"], "actor": tweet["link"].split("/")[3]}
        """
        
        if s["tweet_id"] in tweet_id2categories:
            if len(tweet_id2categories[s["tweet_id"]]) > 1:
                multi_label_pos_count += 1
            pos_count += 1
            s["label"] = list(tweet_id2categories[s["tweet_id"]])
            vocab += s["tokenized_text"]
            """
            if "Other" in s["label"]:
                for i in range(0, 5):
                    train_sentences.append(s)
            else:
                train_sentences.append(s)
            """
            train_sentences.append(s)
        #elif scan_important_words(s["original_text"], important_words) == True:
        #    continue
        else:
            if random.uniform(0, 1) < neg_sample_r:
                neg_count += 1
                s["label"] = ["#Other"] # category: Other
                vocab += s["tokenized_text"]
                train_sentences.append(s)
    
    for tweet_id in test_tweet_ids:
        tweet = tweet_id2tweet[tweet_id]
        cleaned_tweet_words = tokenize(tweet, word_flag)
        
        #cleaned_tweet_words = [chunk[1] for chunk in tweet["chunkList"]]

        s = {"original_text": ' '.join(tweet["body"].split()), "tokenized_text": cleaned_tweet_words, 
            "tweet_id": tweet["tweet_id"], "actor": tweet["link"].split("/")[3], "label": ["#Other"]}
        vocab += s["tokenized_text"]
        test_sentences.append(s)

    print "{} multi-label pos train sentences".format(multi_label_pos_count)
    print "{} pos train sentences".format(pos_count)
    print "{} neg train sentences".format(neg_count)
    print "{} test sentences".format(len(test_sentences))

    vocab = list(set(vocab))
    return train_sentences, test_sentences, vocab

def sigmoid(x):
    return 1.0 / (1.0 + math.exp(-x))

def select_categories(valueList, t):
    categories = []
    for i in range(0, len(valueList)):
        if sigmoid(valueList[i]) >= t:
            categories.append(i)
    return categories


def update_tweet_id2categories(true_and_pred_file, labelList, last_tweet_id2categories, t):
    tweet_id2categories = {}
    output_file = open(true_and_pred_file, "r")
    for line in output_file:
        items = line.split("\t")

        #categories = ast.literal_eval(items[2])

        valueList = ast.literal_eval(items[3])
        categories = select_categories(valueList, t)

        if labelList.index("#Other") in categories:
            categories.remove(labelList.index("#Other"))

        if len(categories) == 0:
            continue
        
        # initialize
        if items[1] not in tweet_id2categories:
            tweet_id2categories[items[1]] = set()

        for c in categories:
            tweet_id2categories[items[1]].add(labelList[c])

    output_file.close()
    
    #tweet_id2categories.update(last_tweet_id2categories)
    
    # only add new labels
    

    for tweet_id in last_tweet_id2categories:
        if tweet_id in tweet_id2categories:
            for old_c in last_tweet_id2categories[tweet_id]:
                tweet_id2categories[tweet_id].add(old_c)
        else:
            tweet_id2categories[tweet_id] = last_tweet_id2categories[tweet_id]

    new_pred_count = 0
    for tweet_id in tweet_id2categories:
        if tweet_id in last_tweet_id2categories:
            new_pred_count += len(tweet_id2categories[tweet_id] - last_tweet_id2categories[tweet_id])
        else:
            new_pred_count += len(tweet_id2categories[tweet_id])

    return tweet_id2categories, new_pred_count
