import sys
sys.path.append("../utilities/")
reload(sys)
sys.setdefaultencoding('utf-8')
import random
import numpy as np
from numpy import dot
from numpy.linalg import norm
import networkx as nx
from collections import defaultdict
import preprocessor as p
import ast, pickle, time, os, math, copy
from multiprocessing import Process
from stanfordcorenlp import StanfordCoreNLP
from utilities import load_event_ontology, tweet_filter, load_important_words, process_text, load_twitter_stopwords, postedTime2seconds
from slpa_utilities import Graph, MetaGraph, MetaGraph2Graph, get_acceptedLabel, initialize_memory, find_communities, graph_density


def lemmatize(nlp, chunk):
    sentence = chunk.replace(u"_", u" ")
    r_dict = nlp._request('ssplit, lemma', sentence)
    tokens = [token['lemma'] for s in r_dict['sentences'] for token in s['tokens']]
    return u"_".join(tokens)




def prepare_data(folder, input_name2hourList, phase_category2keywords, output_flag):

    

    print "prepare_data()..."
    tweet_id2content = {}
    tweet_id_set = set()
    chunk2importance = {}
    chunk2IDF = {}
    
    #hashtag2tweet_num = {}
    mention2tweet_num = {}

    actor2tweet_num = {}
    actor2tweet_ids = {}
    actor2replied_actor = {}
    actor2geo = {}

    

    nlp = StanfordCoreNLP("../../tools/stanford-corenlp-full-2017-06-09")

    word2lemma = {}

    for input_name in input_name2hourList:

        input_file = open(folder + input_name, "r")
        hourList = input_name2hourList[input_name]

        for line in input_file:
            tweet = ast.literal_eval(line)

            if "body" in tweet and "actor_Username" in tweet:
                actor = "@" + tweet["actor_Username"]
                words = tweet["body"].split()
                if words[0][0] == "@":
                    if actor not in actor2replied_actor:
                        actor2replied_actor[actor] = [[words[0], tweet["postedTime"]]]
                    else:
                        actor2replied_actor[actor] += [[words[0], tweet["postedTime"]]]
                elif words[0] == "RT":
                    if actor not in actor2replied_actor:
                        actor2replied_actor[actor] = [[words[1].replace(":", ""), tweet["postedTime"]]]
                    else:
                        actor2replied_actor[actor] += [[words[1].replace(":", ""), tweet["postedTime"]]]


            if tweet_filter(tweet, hourList) == True:
                continue

            words = tweet["link"].split("/")
            tweet_id = words[3] + "_" + words[5]

            chunk_set, hashtag_set, mention_set = set(), set(), set()
            tweet_id_set.add(tweet_id)

            actor = "@" + words[3]

            if actor not in actor2tweet_num:
                actor2tweet_num[actor] = 1
            else:
                actor2tweet_num[actor] +=1

            if actor not in actor2tweet_ids:
                actor2tweet_ids[actor] = [tweet_id]
            else:
                actor2tweet_ids[actor] += [tweet_id]

            if "geo" in tweet:
                if actor not in actor2geo:
                    actor2geo[actor] = [tweet["geo"]]
                else:
                    actor2geo[actor] += [tweet["geo"]]
            
            parsed_tweet = p.parse(tweet["body"])
            mentions = parsed_tweet.mentions
            hashtags = parsed_tweet.hashtags

            sentence_chunks = set()
            for chunk in tweet["chunkList"]:
                if chunk[0] == None:
                    continue

                new_chunk = None
                if chunk[1] in word2lemma:
                    new_chunk = word2lemma[chunk[1]]
                else:
                    new_chunk = lemmatize(nlp, chunk[1])
                    word2lemma[chunk[1]] = new_chunk

                if new_chunk not in chunk2importance:
                    chunk2importance[new_chunk] = list([chunk[2]])
                else:
                    chunk2importance[new_chunk] += [chunk[2]]

                sentence_chunks.add(new_chunk)
                

                chunk_set.add(new_chunk)

            for new_chunk in sentence_chunks:
                if new_chunk not in chunk2IDF:
                    chunk2IDF[new_chunk] = 1.0
                else:
                    chunk2IDF[new_chunk] += 1.0

            if hashtags != None:
                for hashtag in hashtags:
                    tag = hashtag.match.lower()
                    hashtag_set.add(tag)
                    #if tag not in hashtag2tweet_num:
                    #    hashtag2tweet_num[tag] = 1
                    #else:
                    #    hashtag2tweet_num[tag] += 1
            
            if mentions != None:
                for mention in mentions:
                    m = mention.match
                    mention_set.add(m)
                    if m not in mention2tweet_num:
                        mention2tweet_num[m] = 1
                    else:
                        mention2tweet_num[m] += 1

            

            if "geo" in tweet:
                tweet_id2content[tweet_id] = {"body": tweet["body"], "actor": actor, "chunks": chunk_set, "hashtags": hashtag_set, 
                                            "mentions": mention_set, "geo": tweet["geo"], "postedTime": tweet["postedTime"]}
            else:
                tweet_id2content[tweet_id] = {"body": tweet["body"], "actor": actor, "chunks": chunk_set, "hashtags": hashtag_set, 
                                            "mentions": mention_set, "postedTime": tweet["postedTime"]}
        
        input_file.close()
    
    nlp.close()

    total_doc = len(tweet_id_set)
    for chunk in chunk2IDF:
        chunk2IDF[chunk] = math.log(total_doc / chunk2IDF[chunk])

    pickle.dump(tweet_id_set, open(output_flag + "tweet_id_set.p", "wb"))
    pickle.dump(chunk2importance, open(output_flag + "chunk2importance.p", "wb"))
    pickle.dump(chunk2IDF, open(output_flag + "chunk2IDF.p", "wb"))
    pickle.dump(tweet_id2content, open(output_flag + "tweet_id2content.p", "wb"))

    #pickle.dump(hashtag2tweet_num, open(output_flag + "hashtag2tweet_num.p", "wb"))
    pickle.dump(mention2tweet_num, open(output_flag + "mention2tweet_num.p", "wb"))
    pickle.dump(actor2tweet_num, open(output_flag + "actor2tweet_num.p", "wb"))
    pickle.dump(actor2tweet_ids, open(output_flag + "actor2tweet_ids.p", "wb"))
    pickle.dump(actor2replied_actor, open(output_flag + "actor2replied_actor.p", "wb"))
    pickle.dump(actor2geo, open(output_flag + "actor2geo.p", "wb"))

    for phase_category in phase_category2keywords:
        keywords = set(phase_category2keywords[phase_category])
        category_tweet_id_set = set()
        
        for tweet_id in tweet_id2content:
            #if len(tweet_id2content[tweet_id]["chunks"] & keywords) != 0:
            cleaned_tweet_words = p.clean(tweet_id2content[tweet_id]["body"])
            cleaned_tweet_words = process_text(cleaned_tweet_words)
            cleaned_tweet_words = set(cleaned_tweet_words.split())

            if len(cleaned_tweet_words & keywords) != 0:
                category_tweet_id_set.add(tweet_id)
        
        pickle.dump(category_tweet_id_set, open(phase_category + "/tweet_id_set.p", "wb"))
    



def prepare_seed_tweets(folder, input_name, hourList, keyword2phase_category):

    input_file = open(folder + input_name, "r")

    print "processing seed tweets..."

    seed_tweet_id2labels = {}
    keywords = set()
    for keyword in keyword2phase_category:
        keywords.add(keyword)

    for line in input_file:
        tweet = ast.literal_eval(line)
        
        if tweet_filter(tweet, hourList) == True:
            continue
        
        words = tweet["link"].split("/")
        tweet_id = words[3] + "_" + words[5]

        actor = "@" + words[3]

        """
        parsed_tweet = p.parse(tweet["body"])
        mentions = parsed_tweet.mentions
        hashtags = parsed_tweet.hashtags
        """
        tweet_text = process_text(tweet["body"]).lower()

        matched_keywords = keywords & set(tweet_text.split())

        if len(matched_keywords) != 0:
            seed_tweet_id2labels[tweet_id] = set()
            for matched_keyword in matched_keywords:
                seed_tweet_id2labels[tweet_id].add(keyword2phase_category[matched_keyword])
                    
    input_file.close()

    pickle.dump(seed_tweet_id2labels, open("seed_tweet_id2labels.p", "wb"))

def typeList2weight(typeList, type2w):
    sum_w = 0

    count = typeList.count("I_G_w")
    if count != 0:
        w = 2.0 ** (count - 2.0) * type2w["I_G_w"]
        sum_w += w

    count = typeList.count("W_G_w")
    if count >= 2:
        w = 2.0 ** (count - 2.0) * type2w["W_G_w"]
        sum_w += w

    count = typeList.count("G_w")
    if count >= 2:
        w = 2.0 ** (count - 2.0) * type2w["G_w"]
        sum_w += w
    
    return sum_w
    

def get_weight_chunk(tweet1, tweet2, important_words, invalid_chunks):
    
    common_words = tweet1["chunks"] & tweet2["chunks"] - invalid_chunks
    typeList = []
    if len(common_words) == 0:
        return []
    for word in common_words:
        if len(word) <= 3 and word not in important_words:
            continue
        if word in important_words:
            #typeList.append("I_G_w")
            typeList.append("G_w")
        elif word in weak_words:
            typeList.append("W_G_w")
        else:
            typeList.append("G_w")
    return typeList


def get_common_mentions(tweet1, tweet2, time_window):
    if len(tweet1["mentions"]) == 0 or len(tweet2["mentions"]) == 0:
        return []
    else:
        time1 = postedTime2seconds(tweet1["postedTime"])
        time2 = postedTime2seconds(tweet2["postedTime"])
        if abs(time1 - time2) < time_window:
            return list(tweet1["mentions"] & tweet2["mentions"] - set(['@']))
    return []

def get_common_hashtags(tweet1, tweet2):
    return list(tweet1["hashtags"] & tweet2["hashtags"] - set(['#']))


def measure_sim_TFIDF(tweet1, tweet2, invalid_chunks, weak_words, chunk2IDF, chunkList):
    u = np.zeros(len(chunkList))
    v = np.zeros(len(chunkList))

    common_words = tweet1["chunks"] & tweet2["chunks"] - invalid_chunks - weak_words

    if len(common_words) <= 1:
        return None, []

    for chunk in (tweet1["chunks"] - invalid_chunks - weak_words):
        u[chunkList.index(chunk)] += chunk2IDF[chunk]

    for chunk in (tweet2["chunks"] - invalid_chunks - weak_words):
        v[chunkList.index(chunk)] += chunk2IDF[chunk]

    cos_sim = dot(u, v) / (norm(u) * norm(v))

    if cos_sim <= 0.0001:
        return None, []
    else:
        return cos_sim, []

def measure_sim_charLen(tweet1, tweet2, invalid_chunks, weak_words, type2w):
    W = 0
    common_words = tweet1["chunks"] & tweet2["chunks"] - invalid_chunks - weak_words

    if len(common_words) <= 1:
        return None, []

    tweet1_len = len(tweet1["chunks"])
    tweet2_len = len(tweet2["chunks"])

    for common_word in common_words:
        # The average length of English words is 4.5 letters
        W += float(len(common_word)) / 4.5

    W = type2w["G_w"] * 2.0 ** (W - 2.0)

    W = W / (float(max(tweet1_len, tweet2_len)) ** 2) * 5.0 * 5.0

    if W == 0:
        return None, list(common_words)
    else:
        return W, list(common_words)


def measure_sim(tweet1, tweet2, invalid_chunks, important_words, weak_words, \
    actor2tweet_num, actor2tweet_ids, actor2type, mention2tweet_num, mention_avg, type2w):
    
    typeList = []

    W = 0

    
    tempList = get_weight_chunk(tweet1, tweet2, important_words, invalid_chunks)
    typeList += tempList
    tweet1_len = len(tweet1["chunks"])
    tweet2_len = len(tweet2["chunks"])
    if tweet1_len != 0 and tweet2_len != 0:
        W += typeList2weight(tempList, type2w) / float(max(tweet1_len, tweet2_len)) ** 2 * 5.0 * 5.0

    if len(typeList) == 0 or W == 0:
        return None, []
    else:
        return W, typeList


# get all neighbors' similarity score for tweet i (multiprocessing)
def get_neighbors_sim(idx, tweet_idList, tweet_id2content, invalid_chunks, important_words, weak_words, \
    actor2tweet_num, actor2tweet_ids, mention2tweet_num, type2w, chunk2IDF, chunkList):
    ij2sim = {}
    output = open(str(idx) + "_tweet_pairs.txt", "w", 0)
    out_count = 0

    mention_sum = 0
    mention_count = 0
    for mention in mention2tweet_num:
        #if mention2tweet_num[mention] <= 1:
        #    continue
        mention_sum += mention2tweet_num[mention]
        mention_count += 1
    mention_avg = float(mention_sum) / float(mention_count)

    #hashtag_sum = 0
    #for hashtag in hashtag2tweet_num:
    #    hashtag_sum += hashtag2tweet_num[hashtag]
    #hashtag_avg = float(hashtag_sum) / float(len(hashtag2tweet_num))

    pronouns = set(["i", "I", "my", "me", "myself"])
    actor2type = {}
    for actor in actor2tweet_ids:
        me_now = 0
        for tweet_id in actor2tweet_ids[actor]:
            if tweet_id in tweet_id2content:
                words = process_text(tweet_id2content[tweet_id]["body"]).split()
                if len(set(words) & pronouns) >= 1 and "http" not in words[-1]:
                    me_now += 1
        if float(me_now) / float(actor2tweet_num[actor]) >= 0.5:
            actor2type[actor] = "Meformer"
        else:
            actor2type[actor] = "Informer"

    for i in range(0, len(tweet_idList)):
        if i % 20 != idx:
            continue
        tweet1 = tweet_id2content[tweet_idList[i]]
        for j in range(i+1, len(tweet_idList)):
            tweet2 = tweet_id2content[tweet_idList[j]]

            sim, typeList = measure_sim_charLen(tweet1, tweet2, invalid_chunks, weak_words, type2w)

            if sim != None:
                ij2sim[str(i) + ' ' + str(j)] = sim
            if out_count < 10000 and sim != None and "I_G_w" not in typeList and random.uniform(0, 1) < 0.1:
            #if out_count < 6000 and sim != None and "Geo_w" in typeList:
                out_count += 1
                output.write("############################\n")
                output.write(str(tweet1) + "\n")
                output.write(str(tweet2) + "\n")
                output.write("****************************\n")
                output.write(str(sim) + " " + str(typeList) + "\n")
                output.write("############################\n\n\n")
    pickle.dump(ij2sim, open("ij2sim_" + str(idx) + ".p", "wb"))
    output.close()

def tweets_time_window(tweet_id2content, tweet_ids, time_start, time_end):
    contentList = []
    for tweet_id in tweet_ids:
        postedTime = tweet_id2content[tweet_id]["postedTime"]
        time = postedTime2seconds(postedTime)
        if time_start <= time and time <= time_end:
            contentList.append(tweet_id2content[tweet_id])
    return contentList


def create_semantic_graph(target_dir, invalid_chunks, important_words, weak_words, type2w):

    tweet_id_set = pickle.load(open("tweet_id_set.p", "rb"))
    chunk2importance = pickle.load(open(target_dir + "chunk2importance.p", "rb"))
    
    chunk2IDF = pickle.load(open(target_dir + "chunk2IDF.p", "rb"))
    chunkList = chunk2IDF.keys()
    
    tweet_id2content = pickle.load(open(target_dir + "tweet_id2content.p", "rb"))

    actor2tweet_num = pickle.load(open(target_dir + "actor2tweet_num.p", "rb"))
    actor2tweet_ids = pickle.load(open(target_dir + "actor2tweet_ids.p", "rb"))
    #hashtag2tweet_num = pickle.load(open("hashtag2tweet_num.p", "rb"))
    mention2tweet_num = pickle.load(open(target_dir + "mention2tweet_num.p", "rb"))

    chunk2mean_importance = {}
    for chunk in chunk2importance:
        chunk2mean_importance[chunk] = float(sum(chunk2importance[chunk])) / float(len(chunk2importance[chunk])) #* math.log10(float(len(chunk2importance[chunk])))

    sorted_chunk2mean_importance = sorted(chunk2mean_importance.items(), key = lambda e: e[1], reverse = True)
    tweet_idList = list(tweet_id_set)

    # chunk_w, hashtag_w, mention_w, reply_w, context_w 

    G = Graph()
    print "building chunk graph..."

    
    # for all pairs in tweets, build the connection between them
    processV = []
    for idx in range(0, 20):
        processV.append(Process(target = get_neighbors_sim, 
            args = (idx, tweet_idList, tweet_id2content, invalid_chunks, \
                important_words, weak_words, actor2tweet_num, actor2tweet_ids, \
                mention2tweet_num, type2w, chunk2IDF, chunkList, )))
    for idx in range(0, 20):
        processV[idx].start()
    for idx in range(0, 20):
        processV[idx].join()
    

    for idx in range(0, 20):
        #print "ij2sim_" + str(idx) + ".p"
        ij2sim = pickle.load(open("ij2sim_" + str(idx) + ".p", "rb"))
        for ij in ij2sim:
            sim = ij2sim[ij]
            if sim != None and sim != 0:
                i = int(ij.split()[0])
                j = int(ij.split()[1])
                
                G.add_edge(i, j, sim)
                G.add_edge(j, i, sim)
        #os.system("rm " + "ij2sim_" + str(idx) + ".p")

    print "graph_density(G):", graph_density(G)

    return G


if __name__ == "__main__":
    random.seed(11)

    phase_category2keywords = load_event_ontology("../dic/event_ontology_new.txt")

    phase_category2keywords = phase_category2keywords["Impact"]

    keyword2phase_category = {}
    
    for category in phase_category2keywords:
        for keyword in phase_category2keywords[category]:
            keyword2phase_category[keyword] = category

    folder = "../run_SNLI_encoder4/"
    
    input_name2hourList = {"2017_08_28_tweetList.txt": ["06", "07", "08", "09", "10", "11", "12", "13", "14", "15", "16", "17"]}
    
    #input_name2hourList = {"2017_08_25_tweetList.txt": ["11", "12", "13", "14", "15", "16", "17", "18", "19", "20", "21", "22", "23"], "2017_08_26_tweetList.txt": ["00", "01", "02", "03", "04", "05", "06", "07", "08", "09", "10"]}
    
    #input_name2hourList = {"2017_08_26_tweetList.txt": ["11", "12", "13", "14", "15", "16", "17", "18", "19", "20", "21", "22", "23"], "2017_08_27_tweetList.txt": ["00", "01", "02", "03", "04", "05", "06", "07", "08", "09", "10"]}
    
    #input_name2hourList = {"2017_08_27_tweetList.txt": ["11", "12", "13", "14", "15", "16", "17", "18", "19", "20", "21", "22", "23"], "2017_08_28_tweetList.txt": ["00", "01", "02", "03", "04", "05", "06", "07", "08", "09", "10"]}
    
    #input_name2hourList = {"2017_08_28_tweetList.txt": ["11", "12", "13", "14", "15", "16", "17", "18", "19", "20", "21", "22", "23"], "2017_08_29_tweetList.txt": ["00", "01", "02", "03", "04", "05", "06", "07", "08", "09", "10"]}
    
    #input_name2hourList = {"2017_08_29_tweetList.txt": ["11", "12", "13", "14", "15", "16", "17", "18", "19", "20", "21", "22", "23"], "2017_08_30_tweetList.txt": ["00", "01", "02", "03", "04", "05", "06", "07", "08", "09", "10"]}
    
    #input_name2hourList = {"2017_08_30_tweetList.txt": ["11", "12", "13", "14", "15", "16", "17", "18", "19", "20", "21", "22", "23"], "2017_08_31_tweetList.txt": ["00", "01", "02", "03", "04", "05", "06", "07", "08", "09", "10"]}
    
    #input_name2hourList = {"2017_08_31_tweetList.txt": ["11", "12", "13", "14", "15", "16", "17", "18", "19", "20", "21", "22", "23"], "2017_09_01_tweetList.txt": ["00", "01", "02", "03", "04", "05", "06", "07", "08", "09", "10"]}
    
    #input_name2hourList = {"2017_09_01_tweetList.txt": ["11", "12", "13", "14", "15", "16", "17", "18", "19", "20", "21", "22", "23"], "2017_09_02_tweetList.txt": ["00", "01", "02", "03", "04", "05", "06", "07", "08", "09", "10"]}
    

    """
    input_name = "Florence_2018_09_17_tweetList.txt"
    hourList = ["06", "07", "08", "09", "10", "11", "12", "13", "14", "15", "16", "17", "18"]
    """
    invalid_chunks = set(["post", "retweet", "rt", "gt", "amp", "lol", "wan", "na", "yeah", "fucking", "fuck", "damn", "ohhh", "ohh", \
        "gon", "na", "lmao", None])

    twitter_stop_words = load_twitter_stopwords("../dic/twitter-stopwords-TA-Less.txt")

    invalid_chunks = invalid_chunks | twitter_stop_words


    important_words = set()
    for category in phase_category2keywords:
        important_words = important_words | set(phase_category2keywords[category])


    weak_words = set(["thank", "thanks", "houston", "texas", "carolina", "please", "city", "days", \
        "flood", "flooded", "flooding", "people", "harvey", "florence", "florance", "hurricane", \
        "storm", "storms", "weather", "carolinas", "north", "south", "disaster"])



    for phase_category in phase_category2keywords:
        if not os.path.exists(phase_category):
            os.makedirs(phase_category)

    prepare_data(folder, input_name2hourList, phase_category2keywords, output_flag = "")
    #prepare_seed_tweets(folder, input_name, hourList, keyword2phase_category)

    for phase_category in phase_category2keywords:
        print "#############", phase_category, "##############"
        
        os.chdir(phase_category)

        # G_w: unigram_w, I_G_w: important_unigram, W_G_w: weak_unigram_w, H_w: hashtag_w, M_w: mention_w
        # R_w: reply_w, C_w: context_w, Geo_w: geo location weight
        type2w = {"G_w":0.05, "I_G_w":0.20, "W_G_w":0, "H_w":0.1, "M_w":0.05, "R_w":0.1, "C_w":0.05, "Geo_w":0.02}
        
        G = create_semantic_graph("../", invalid_chunks, important_words, weak_words, type2w)

        
        tweet_id_set = pickle.load(open("tweet_id_set.p", "rb"))
        tweet_idList = list(tweet_id_set)

        tweet_id2idx = {}
        for i, tweet_id in enumerate(tweet_idList):
            tweet_id2idx[tweet_id] = i
        
        
        T = 60
        r = 0.40
        #decay_r = 0.95
        decay_r = 1.0

        seed_tweet_idx2labels = {}
        print "running community detection (pass 1)..."
        memory = initialize_memory(G, seed_tweet_idx2labels, "1")
        communities = find_communities(G, T, r, decay_r, memory)
        pickle.dump(communities, open("communities_pass1.p", "wb"))

        os.chdir("../") # go back to parent dir
"""
if __name__ == "__main__":
    nlp = StanfordCoreNLP("../../tools/stanford-corenlp-full-2017-06-09")
    new_chunk = lemmatize(nlp, "reservoirs")
    print new_chunk
"""
