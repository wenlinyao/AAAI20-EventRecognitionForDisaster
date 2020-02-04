import sys
sys.path.append("../utilities/")
reload(sys)
sys.setdefaultencoding('utf-8')
import os
import json
import gzip
import operator
import string
import pickle
import preprocessor as p
from multiprocessing import Process
import gensim
from gensim import corpora
from nltk.corpus import stopwords 
from nltk.stem.wordnet import WordNetLemmatizer
from utilities import get_timeList, load_stopwords, load_event_ontology, extract_tweet_info



def process_files(folder, dateList, hourList, minuteList):
    tweet_id2tweet = {}
    all_tweet_ids = set()
    original_tweet_id2reply_ids = {}

    for valid_date in dateList:
        print valid_date
        #if not os.path.exists(valid_date):
        #    os.makedirs(valid_date)
        
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
                    if "body" not in tweet or "link" not in tweet:
                        continue
                    if tweet["body"].split()[0] == "RT":
                        continue
                    
                    words = tweet["link"].split("/")
                    tweet_id = words[3] + "_" + words[5]
                    tweet_id2tweet[tweet_id] = new_tweet

                    if "inReplyTo" in tweet:
                        words = tweet["inReplyTo"]["link"].split("/")
                        original_tweet_id = words[3] + "_" + words[5]
                        if original_tweet_id not in original_tweet_id2reply_ids:
                            original_tweet_id2reply_ids[original_tweet_id] = [tweet_id]
                        else:
                            original_tweet_id2reply_ids[original_tweet_id] += [tweet_id]
                    else:
                        all_tweet_ids.add(tweet_id)

                input_file.close()
    output = open("reply_tweets.txt", "w")
    for tweet_id in all_tweet_ids:
        if tweet_id not in tweet_id2tweet:
            continue
        if tweet_id not in original_tweet_id2reply_ids:
            continue
        #if len(original_tweet_id2reply_ids[tweet_id]) < 5:
        #    continue
        output.write(str(tweet_id2tweet[tweet_id]) + "\n")
        for reply_id in original_tweet_id2reply_ids[tweet_id]:
            if reply_id in tweet_id2tweet:
                output.write(str(tweet_id2tweet[reply_id]) + "\n")
        output.write("*****************************************\n\n\n")
    output.close()
    
    pickle.dump(original_tweet_id2reply_ids, open("original_tweet_id2reply_ids.p", "wb"))
    print float(len(original_tweet_id2reply_ids)) / float(len(all_tweet_ids))
    reply_numList = [len(original_tweet_id2reply_ids[tweet_id]) for tweet_id in original_tweet_id2reply_ids]
    print float(sum(reply_numList)) / float(len(reply_numList))
    


if __name__ == "__main__":
    folder = "/data/GNIPHarveyTweets/"
    #folder = "../data/"

    dateList, hourList, minuteList = get_timeList()

    dateList = ["2017_08_25", "2017_08_26", "2017_08_27", "2017_08_28", "2017_08_29", "2017_08_30", "2017_08_31"]

    #phase_category2keywords = load_event_ontology("../dic/event_ontology.txt")

    process_files(folder, dateList, hourList, minuteList)
    

    print "finished!"