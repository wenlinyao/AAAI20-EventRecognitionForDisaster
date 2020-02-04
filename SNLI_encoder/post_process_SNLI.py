# add new attributes

import sys
sys.path.append("../utilities/")
reload(sys)
sys.setdefaultencoding('utf-8')
import os, json, gzip, operator, string, pickle, time, ast
import numpy as np
import preprocessor as p
from multiprocessing import Process
from utilities import get_timeList, extract_tweet_info


def get_link2chunkList(folder, date):
    link2chunkList = {}
    input_file = open(folder + date + "_tweetList.txt", "r")
    for line in input_file:
        tweet = ast.literal_eval(line)
        link2chunkList[tweet["link"]] = tweet["chunkList"]
    input_file.close()
    return link2chunkList


def process_files(folder, i, valid_date, hourList, minuteList):
    print i, valid_date
    link2chunkList = get_link2chunkList("../run_SNLI_encoder3/", valid_date)

    output = open(valid_date + "_tweetList.txt", "w")

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
                
                if "body" in new_tweet and new_tweet["link"] in link2chunkList:
                    new_tweet["chunkList"] = link2chunkList[new_tweet["link"]]
                    output.write(str(new_tweet) + "\n")

            input_file.close()
    output.close()
    



if __name__ == "__main__":
    
    folder = "/data/GNIPHarveyTweets/"
    #folder = "../data/"

    dateList, hourList, minuteList = get_timeList()
    dateList = ["2017_08_27", "2017_08_28"]

    for j in range(0, 4):
        new_dateList = dateList[j*3:(j+1)*3]

        processV = []
        for i in range(0, len(new_dateList)):
            processV.append(Process(target = process_files, args = (folder, i, new_dateList[i], hourList, minuteList,)))
        for i in range(0, len(new_dateList)):
            processV[i].start()
        for i in range(0, len(new_dateList)):
            processV[i].join()
