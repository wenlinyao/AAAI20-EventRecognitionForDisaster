import sys
sys.path.append("../utilities/")
reload(sys)
sys.setdefaultencoding('utf-8')
import ast, glob, math, pickle, random, os
from utilities import load_event_ontology
from nltk.stem.wordnet import WordNetLemmatizer

def entropy(pList):
    normalize_pList = []
    for p in pList:
        normalize_pList.append(p / sum(pList))
    ent = 0
    for p in normalize_pList:
        if p != 0:
            ent -= p * math.log(p, 2)
    return ent

def find_category(sorted_chunk2freq, import_word2category):
    categoryList = []
    for item in sorted_chunk2freq[:6]:
        if item[0] in import_word2category:
            categoryList.append(import_word2category[item[0]])
    return categoryList

def find_major_category(categoryList):
    if len(categoryList) == 0:
        return "#Other"
    category2freq = {}
    for category in categoryList:
        if category not in category2freq:
            category2freq[category] = 1
        else:
            category2freq[category] += 1
    sorted_category2freq = sorted(category2freq.items(), key = lambda e: e[1], reverse = True)
    return sorted_category2freq[0][0]


if __name__ == "__main__":
    random.seed(11)

    #               0                 1                      2               3                  4                        5       
    #labelList = ["#Other", "#Preventative_measure", "#Help_and_rescue", "#Casualty", "#Utilities_and_Supplies", "#Transportation",
    #                           6                               7
    #            "#Flood_control_infrastructures", "#Built-environment_hazards"]

    #                       1                      2               3            4    
    labelList = ["Preventative_measure", "Help_and_rescue", "Casualty", "Housing", 
    #                       5                        6                         7
                "Utilities_and_Supplies", "Transportation", "Flood_control_infrastructures", 
    #                      8                          9                   10
                "Business_Work_School", "Built-environment_hazards", "Other"]



    lmtzr = WordNetLemmatizer()
    invalid_words = set(["rt", "gt", "amp", "lol", "wan_na", "yeah", "fucking", "fuck", "damn", "ohhh", "ohh", "gon_na", "lmao"])
    

    #folder = "../run_model_slpa_Word_v3_8/communities_59/"


    phase_category2keywords = load_event_ontology("../dic/event_ontology_new.txt")

    category2important_words = phase_category2keywords["Impact"]

    important_words = set()
    import_word2category = {}
    
    for category in category2important_words:
        for word in category2important_words[category]:
            import_word2category[word] = category
        important_words = important_words | set(category2important_words[category])

    print category2important_words

    """
    category2important_words = load_important_words("../dic/important_words.txt")
    #category2important_words = load_important_words("../dic/important_words_after.txt")

    important_words = set()
    import_word2category = {}
    for category in category2important_words:
        for word in category2important_words[category]:
            import_word2category[word] = category
        important_words = important_words | category2important_words[category]

    print category2important_words
    """
    target_folder = "../run_model_slpa_Word_v4_2/"
    target_folder = "../run_model_slpa_Word_v4_3/"
    target_folder = "../run_model_slpa_Word_v4_4/"
    target_folder = "../run_model_slpa_Word_v4_5/"
    target_folder = "../run_model_slpa_Word_v4_TFIDF/"
    target_folder = "../run_model_slpa_Word_v4_TFIDF2/"
    target_folder = "../run_model_slpa_Word_v4_6/" # use charLen similarity
    target_folder = "../run_model_slpa_Word_v4_7/" # use charLen similarity, W = type2w["G_w"] * 2.0 ** (W - 2.0)
    target_folder = "../run_model_slpa_Word_v4_8/" # use charLen similarity, W = type2w["G_w"] * 2.0 ** (W - 2.0), + died, dies
    target_folder = "../run_model_slpa_Word_v4_Day30/"
    target_folder = "../run_model_slpa_Word_v4_Florence_2/"
    #target_folder = "../run_model_slpa_Word_v4_8_fix/" # only use 12 hours data in SLPA
    target_folderList = ["../run_model_slpa_Word_v4_0825-0826/", "../run_model_slpa_Word_v4_0826-0827/", "../run_model_slpa_Word_v4_0827-0828/",\
                        "../run_model_slpa_Word_v4_0828-0829/", "../run_model_slpa_Word_v4_0829-0830/", "../run_model_slpa_Word_v4_0830-0831/",\
                        "../run_model_slpa_Word_v4_0831-0901/", "../run_model_slpa_Word_v4_0901-0902/"]

    target_folderList = ["../run_model_slpa_Word_v4_PostClassification/"]

    target_folderList = ["../run_model_slpa_Word_v4_PostClassification_0825-0826/", "../run_model_slpa_Word_v4_PostClassification_0826-0827/", \
                        "../run_model_slpa_Word_v4_PostClassification_0827-0828/", "../run_model_slpa_Word_v4_PostClassification_0828-0829/", \
                        "../run_model_slpa_Word_v4_PostClassification_0829-0830/", "../run_model_slpa_Word_v4_PostClassification_0830-0831/",]
    
    for target_folder in target_folderList:
        write_folder = target_folder.split("/")[1] + "/"
        if not os.path.exists(write_folder):
            os.makedirs(write_folder)

        for label in labelList:
            if label == "Other":
                continue
            folder = target_folder + label + "/communities_59/"
            output = open(write_folder + label + "_communities_chunks.txt", "w")

            print folder
            file2len = {}
            for file in glob.glob(folder + "*.txt"):
                file2len[file] = len(open(file, "r").readlines())
            sorted_file2len = sorted(file2len.items(), key = lambda e: e[1], reverse = True)

            top_word2category = {}

            clusterId2top_chunks = {}

            tweet_id2labels = {}
            for item in sorted_file2len:
                
                #if "675.txt" not in file and "508.txt" not in file:
                #   continue
                file = item[0]
                input_file = open(file, "r")

                chunk2freq = {}
                hashtag2freq = {}

                count = 0

                tweet_ids = []
                tweetList = []
                print file
                for line in input_file:
                    if not line.strip():
                        continue
                    #print line
                    tweet = ast.literal_eval(line)
                    
                    if "chunkList" not in tweet:
                        continue
                    tweetList.append(tweet)
                    
                    words = tweet["link"].split("/")
                    tweet_id = words[3] + "_" + words[5]
                    tweet_ids.append(tweet_id)

                    for chunk in tweet["chunkList"]:
                        if chunk[1] in invalid_words:
                           continue
                        if chunk[1] not in chunk2freq:
                            chunk2freq[chunk[1]] = 1
                        else:
                            chunk2freq[chunk[1]] += 1
                    if "body" in tweet:
                        count += 1
                        for word in tweet["body"].split():
                            if word[0] == "#":
                                word = word.lower()
                                if word not in hashtag2freq:
                                    hashtag2freq[word] = 1
                                else:
                                    hashtag2freq[word] += 1
                #if count < 10:
                #    continue

                for chunk in chunk2freq:
                    p = float(chunk2freq[chunk]) / float(count)
                    chunk2freq[chunk] = [chunk2freq[chunk], float("%0.3f" % p)]
                for hashtag in hashtag2freq:
                    p = float(hashtag2freq[hashtag]) / float(count)
                    hashtag2freq[hashtag] = [hashtag2freq[hashtag], float("%0.3f" % p)]
                
                sorted_chunk2freq = sorted(chunk2freq.items(), key = lambda e: e[1][1], reverse = True)

                categoryList = find_category(sorted_chunk2freq, import_word2category)

                if len(categoryList) != 0:
                    for item in sorted_chunk2freq[:10]:
                        top_word = lmtzr.lemmatize(item[0])
                        #if top_word in invalid_words:
                        #    continue
                        if top_word not in top_word2category:
                            top_word2category[top_word] = [[categoryList[0], item[1][0], item[1][1]]]
                        else:
                            flag = False
                            for i in range(len(top_word2category[top_word])):
                                if categoryList[0] == top_word2category[top_word][i][0]:
                                    top_word2category[top_word][i][1] += item[1][0]
                                    top_word2category[top_word][i][2] += item[1][1]
                                    flag = True
                            if flag == False:
                                top_word2category[top_word] += [[categoryList[0], item[1][0], item[1][1]]]
                
                # update tweet_id2labels
                if len(categoryList) != 0:
                    major_category = labelList.index(find_major_category(categoryList))
                    for tweet_id in tweet_ids:
                        if tweet_id not in tweet_id2labels:
                            tweet_id2labels[tweet_id] = set()
                            tweet_id2labels[tweet_id].add(major_category)
                        else:
                            tweet_id2labels[tweet_id].add(major_category)
                else:
                    major_category = labelList.index("Other")
                    for tweet_id in tweet_ids:
                        tweet_id2labels[tweet_id] = set([major_category])


                output.write(file + " | total: " + str(count) + " | " + str(categoryList) + "\n")
                output.write(file.split("/")[2] + "/" + file.split("/")[4] + "\n")

                random.shuffle(tweetList)
                for tweet in tweetList[:9]:
                    output.write(str(tweet) + "\n\n")

                top_chunks = []
                for item in sorted_chunk2freq:
                    
                    if item[1][1] > 0.02:
                        top_chunks.append(item)
                    if item[1][1] > 0.02:
                        output.write(str(item) + "\n")

                clusterId2top_chunks[file.split("/")[-1]] = top_chunks

                sorted_hashtag2freq = sorted(hashtag2freq.items(), key = lambda e:e[1][1], reverse = True)
                for item in sorted_hashtag2freq:
                    if item[1][1] < 0.02:
                        break
                    output.write(str(item) + "\n")
                input_file.close()
                
                output.write("\n\n\n")



            # save tweets label result
            #pickle.dump(tweet_id2labels, open("tweet_id2labels.p", "wb"))


            # find important_keywords, weak keywords, vague keywords
            top_word2entropy = {}
            for top_word in top_word2category:
                # calculate entropy
                pList = []
                for item in top_word2category[top_word]:
                    pList.append(item[2])
                top_word2entropy[top_word] = [entropy(pList), sum(pList)]

            sorted_top_word2entropy = sorted(top_word2entropy.items(), key = lambda e: (-e[1][0], e[1][1]), reverse = True)

            for item in sorted_top_word2entropy:
                top_word = item[0]
                output.write(top_word + " | " + str(top_word2category[top_word]) + "\n")
            output.write("\n\n")
            

            chunk2clusterIds = {}


            for clusterId in clusterId2top_chunks:
                for top_chunk in clusterId2top_chunks[clusterId]:
                    if top_chunk[0] not in chunk2clusterIds:
                        # top_chunk[0] is the phrase
                        chunk2clusterIds[top_chunk[0]] = [[clusterId, top_chunk[1][0], top_chunk[1][1]]]
                    else:
                        chunk2clusterIds[top_chunk[0]] += [[clusterId, top_chunk[1][0], top_chunk[1][1]]]

            chunk2entropy = {}
            for chunk in chunk2clusterIds:
                pList = []
                for clusterId in chunk2clusterIds[chunk]:
                    pList.append(float(clusterId[2]))
                # total frequency
                if sum([clusterId[1] for clusterId in chunk2clusterIds[chunk]]) > 10:
                    chunk2entropy[chunk] = [entropy(pList), sum(pList)]

            sorted_chunk2entropy = sorted(chunk2entropy.items(), key=lambda e: (e[1][0], -e[1][1]), reverse = True)
            for item in sorted_chunk2entropy:
                chunk = item[0]
                output.write(chunk + " | " + str(chunk2entropy[chunk]) + " | " + str(chunk2clusterIds[chunk]) + "\n")

            output.close()


