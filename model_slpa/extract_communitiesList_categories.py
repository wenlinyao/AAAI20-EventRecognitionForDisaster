import pickle, ast, os, argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', dest='mode', type = str, metavar='<str>', default='', help='test_ or nothing')
    parser.add_argument('--input_nameList', dest='input_nameList', type = str, metavar='<str>', default='', help='e.g., 2017_08_25_tweetList.txt/2017_08_26_tweetList.txt')
    args = parser.parse_args()

    input_nameList = args.input_nameList.split("/")

    #input_nameList = ["2017_08_28_tweetList.txt"]
    #input_nameList = ["2017_08_30_tweetList.txt"]
    #input_nameList = ["Florence_2018_09_17_tweetList.txt"]

    #input_nameList = ["2017_08_25_tweetList.txt", "2017_08_26_tweetList.txt"]
    #input_nameList = ["2017_08_26_tweetList.txt", "2017_08_27_tweetList.txt"]
    #input_nameList = ["2017_08_27_tweetList.txt", "2017_08_28_tweetList.txt"]
    #input_nameList = ["2017_08_28_tweetList.txt", "2017_08_29_tweetList.txt"]
    #input_nameList = ["2017_08_29_tweetList.txt", "2017_08_30_tweetList.txt"]
    #input_nameList = ["2017_08_30_tweetList.txt", "2017_08_31_tweetList.txt"]
    #input_nameList = ["2017_08_31_tweetList.txt", "2017_09_01_tweetList.txt"]
    #input_nameList = ["2017_09_01_tweetList.txt", "2017_09_02_tweetList.txt"]

    folder = "../run_SNLI_encoder4/"

    tweet_id2tweet = {}

    for input_name in input_nameList:
        input_file = open(folder + input_name, "r")

        for line in input_file:
            tweet = ast.literal_eval(line)
            if "body" not in tweet or "chunkList" not in tweet or "link" not in tweet:
                continue
            if tweet["body"].split()[0] == "RT":
                continue
            #if tweet["chunkList"][0][0] == None:
            #    continue
            if len(tweet["chunkList"]) == 0:
                continue
            words = tweet["link"].split("/")
            tweet_id = words[3] + "_" + words[5]
            tweet_id2tweet[tweet_id] = tweet

        input_file.close()

    #seed_tweet_id2labels = pickle.load(open(args.mode + "seed_tweet_id2labels.p", "rb"))

    #                       1                      2               3            4    
    labelList = ["#Preventative_measure", "#Help_and_rescue", "#Casualty", "#Housing", 
    #                       5                        6                         7
                "#Utilities_and_Supplies", "#Transportation", "#Flood_control_infrastructures", 
    #                      8                          9                   10
                "#Business_Work_School", "#Built-environment_hazards"]

    for folder in labelList:
        print folder
        os.chdir(folder.replace("#", ""))

        train_tweet_id_set = pickle.load(open("tweet_id_set.p", "rb"))
        if args.mode == "test_":
            test_tweet_id_set = pickle.load(open("test_tweet_id_set.p", "rb"))
            tweet_ids = list(train_tweet_id_set) + list(test_tweet_id_set)
        else:
            tweet_ids = list(train_tweet_id_set)
    
        print "extract communitiesList tweets..."

        communitiesList = pickle.load(open(args.mode + "communitiesList.p", "rb"))
        memoryList = pickle.load(open(args.mode + "memoryList.p", "rb"))

        #for i, communities in enumerate(communitiesList):
        i = len(communitiesList) - 1
        communities = communitiesList[-1]
        output_folder = args.mode + "communities_" + str(i) + "/"
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
        count = 0
        for key in communities:
            count += 1
            output = open(output_folder + str(count) + "_" + str(key) + ".txt", "w")
            
            for idx in communities[key]:
                if type(idx) == unicode or type(idx) == str:
                    continue
                if args.mode == "test_" and tweet_ids[idx] not in test_tweet_id_set:
                    continue
                if tweet_ids[idx] not in tweet_id2tweet:
                    continue
                output.write(str(tweet_id2tweet[tweet_ids[idx]]) + "\n")
                output.write(str(memoryList[i][idx]) + "\n\n")
                
            output.close()
        os.chdir("../")
    
        