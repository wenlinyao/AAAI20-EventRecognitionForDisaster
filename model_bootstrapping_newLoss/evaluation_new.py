# consider multi-class in evaluation

import sys
sys.path.append("../utilities/")
reload(sys)
sys.setdefaultencoding('utf-8')
from utilities import tweet_filter
import glob, ast, pickle
import numpy as np
from sklearn.metrics import precision_recall_fscore_support

def keep_letters(s):
    return "".join(x for x in s if x.isalpha())


def evaluation_main(tweet_id2categories, tweet_id2tweet, args):

    all_files = []

    """
    folderList = ["../annotation/round4/", "../annotation/round5/", "../annotation/round6/", "../annotation/round7/"]
    #folderList = ["../annotation/round7/"]
    fileList = ["AE.txt", "CF.txt", "QL.txt", "YY.txt"]
    for folder in folderList:
        for file in fileList:
            all_files.append(folder + file)
    

    
    folderList = ["../annotation/round8/"]
    #fileList = ["Cheng.txt", "Shiva.txt"]
    fileList = ["Cheng.txt"]
    #fileList = ["Shiva.txt"]
    for folder in folderList:
        for file in fileList:
            all_files.append(folder + file)
    """
    
    if args.data_source == "Harvey":
        folderList = ["../annotation/round9/", "../annotation/round9_2/"]
        #fileList = ["Cheng.txt", "Shiva.txt"]
        #fileList = ["Cheng.txt"]
        fileList = ["Shiva.txt"]
        for folder in folderList:
            for file in fileList:
                all_files.append(folder + file)

        
        folderList = ["../annotation/round9_master/"]
        #fileList = ["Cheng.txt", "Shiva.txt"]
        fileList = ["Cheng.txt"]
        #fileList = ["Shiva.txt"]
        for folder in folderList:
            for file in fileList:
                all_files.append(folder + file)

        output = open("error_analysis_8_28_13-18.txt", "w")
    
    elif args.data_source == "Florence":
        # !!!!!!!!!!!!!!!!! Florence !!!!!!!!!!!!!!!!!!!
        folderList = ["../annotation/round11_Florence/"]
        fileList = ["round11.txt"]
        for folder in folderList:
            for file in fileList:
                all_files.append(folder + file)
        
        output = open("error_analysis_Florence.txt", "w")
    
    #                       0                        1                      2               3            4    
    #labelList = ["#Natural_environment", "#Preventative_measure", "#Help_and_rescue", "#Casualty", "#Housing", 
    #                       5                        6                         7
    #            "#Utilities_and_Supplies", "#Transportation", "#Flood_control_infrastructures", 
    #                      8                          9                   10
    #            "#Business_Work_School", "#Built-environment_hazards", "#Other"]

    label2new_label = {0:10, 1:1, 2:2, 3:3, 4:4, 5:5, 6:6, 7:7, 8:8, 9:9, 10:10, 11:10}

    label2category = {0: "#Natural_environment", 1: "#Preventative_measure", 2: "#Help_and_rescue", 3: "#Casualty", \
                  4: "#Housing", 5: "#Utilities_and_Supplies", 6: "#Transportation", 7: "#Flood_control_infrastructures", \
                  8: "#Business_Work_School", 9: "#Built-environment_hazards", 10: "#Other"}

    category2label = {label2category[label]: label for label in label2category}

    tweet_id2labels = {}

    for tweet_id in tweet_id2categories:
        tweet_id2labels[tweet_id] = set()
        for category in tweet_id2categories[tweet_id]:
            label = category2label[category]
            tweet_id2labels[tweet_id].add(label2new_label[label])

    gold_tweet_id2labels = {}
    gold_tweet_id2file = {}

    
    for file in all_files:
        #print folder + file
        annotation_file = open(file, "r")
        for line in annotation_file:
            if not line.strip():
                continue
            items = line.split("\t")
            #print line
            if len(items) != 14:
                print "Error:", file, line
                continue
            label = 10

            words = items[1].split("/")
            tweet_id = words[3] + "_" + words[5]
            gold_tweet_id2file[tweet_id] = file

            for i, item in enumerate(items[3:]):
                if len(item.split()) != 0:
                    if "Not English" not in item:
                        label = label2new_label[i]
                    else:
                        label = label2new_label[i+1]
                    
                    if tweet_id not in gold_tweet_id2labels:
                        gold_tweet_id2labels[tweet_id] = set([label])
                    else:
                        gold_tweet_id2labels[tweet_id].add(label)
            #print label,
            tweet = keep_letters(items[0])



    #print gold_tweet_id2label

    annotation_file.close()

    pickle.dump(gold_tweet_id2labels, open("gold-tweet_id2labels.p", "wb"))

    gold_tweet_id2labels = pickle.load(open("gold-tweet_id2labels.p", "rb"))

    # it is generated from LSTM_bootstrapping_main.py
    #tweet_id2tweet = pickle.load(open("tweet_id2tweet.p", "rb"))

    #output = open("error_analysis_8_27_09-14.txt", "w")

    gold_label2tweet_ids = {}
    for tweet_id in gold_tweet_id2labels:
        for l in gold_tweet_id2labels[tweet_id]:
            if l not in gold_label2tweet_ids:
                gold_label2tweet_ids[l] = set([tweet_id])
            else:
                gold_label2tweet_ids[l].add(tweet_id)

    pred_label2tweet_ids = {}
    for tweet_id in tweet_id2labels:
        for l in tweet_id2labels[tweet_id]:
            if l not in pred_label2tweet_ids:
                pred_label2tweet_ids[l] = set([tweet_id])
            else:
                pred_label2tweet_ids[l].add(tweet_id)

    

    all_gold_ids = set(gold_tweet_id2labels.keys())
    all_pred_ids = set(tweet_id2labels.keys())

    print "len(all_gold_ids - all_pred_ids):", len(all_gold_ids - all_pred_ids)


    for label in range(1, 11):
        if label not in gold_label2tweet_ids or label not in pred_label2tweet_ids:
            continue
        output.write("##################### Category: " + label2category[label] + " ######################\n")
        gold_tweet_ids = gold_label2tweet_ids[label]
        pred_tweet_ids = pred_label2tweet_ids[label]
        for tweet_id in (gold_tweet_ids & pred_tweet_ids & all_gold_ids & all_pred_ids):
            output.write("TP " + str(gold_tweet_id2labels[tweet_id]) + " " + str(tweet_id2labels[tweet_id]) + " " + gold_tweet_id2file[tweet_id] + "\n")
            output.write(str(tweet_id2tweet[tweet_id]) + "\n\n")
        output.write("\n\n")

        for tweet_id in (gold_tweet_ids - pred_tweet_ids & all_gold_ids & all_pred_ids):
            output.write("FN " + str(gold_tweet_id2labels[tweet_id]) + " " + str(tweet_id2labels[tweet_id]) + " " + gold_tweet_id2file[tweet_id] + "\n")
            output.write(str(tweet_id2tweet[tweet_id]) + "\n\n")
        output.write("\n\n")

        for tweet_id in (pred_tweet_ids - gold_tweet_ids & all_gold_ids & all_pred_ids):
            output.write("FP " + str(gold_tweet_id2labels[tweet_id]) + " " + str(tweet_id2labels[tweet_id]) + " " + gold_tweet_id2file[tweet_id] + "\n")
            output.write(str(tweet_id2tweet[tweet_id]) + "\n\n")
        output.write("\n\n")

    output.close()

    
    y_pred = []
    y_true = []
    for tweet_id in tweet_id2labels:
        if tweet_id in gold_tweet_id2labels:
            c = [0 for i in range(0, 11)]
            for l in tweet_id2labels[tweet_id]:
                c[l] = 1
            y_pred.append(c)
            c = [0 for i in range(0, 11)]
            for l in gold_tweet_id2labels[tweet_id]:
                c[l] = 1
            y_true.append(c)

    
    assert len(y_pred) == len(y_true)

    
    y_pred = np.array(y_pred)
    y_true = np.array(y_true)

    PList, RList, F_betaList, numList = precision_recall_fscore_support(y_true=y_true, y_pred=y_pred, average = None, labels = range(0, 11))

    PList, RList, F_betaList, numList = PList.tolist(), RList.tolist(), F_betaList.tolist(), numList.tolist()

    print "\n"
    for i in range(1, 11):
        print "Cls :", i, "(", numList[i], ")", "P", PList[i], "R", RList[i], "F1", F_betaList[i]

    result = []

    for i in range(1, 10):
        performance = "%0.1f" % (PList[i]*100) + " " + "%0.1f" % (RList[i]*100) + " " + "%0.1f" % (F_betaList[i]*100)
        print performance,
        result.append(performance)
    
    avg_P = sum(PList[1:-1]) / float(len(PList[1:-1]))
    avg_R = sum(RList[1:-1]) / float(len(RList[1:-1]))
    if avg_P == 0 and avg_R == 0:
        avg_F1 = 0
    else:
        avg_F1 = 2 * avg_P * avg_R / (avg_P + avg_R)

    performance = "%0.1f" % (avg_P*100) + " " + "%0.1f" % (avg_R*100) + " " + "%0.1f" % (avg_F1*100)
    print performance,
    result.append(performance)


    print "\n"
    print PList
    print RList

    return " ".join(result)





