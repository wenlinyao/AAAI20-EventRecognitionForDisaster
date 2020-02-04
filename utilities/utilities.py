import numpy as np
import sklearn, string, re
import preprocessor as p
from nltk.corpus import stopwords 
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import words
from sklearn import metrics
import networkx as nx
#import matplotlib.pyplot as plt
from langdetect import detect_langs
from langdetect.lang_detect_exception import LangDetectException

nltk_words = set(words.words())

class Node():
    def __init__(self, auth, hub):
        self.auth = auth
        self.hub = hub

class Actor():
    def __init__(self, link, followersCount, friendsCount, statusesCount, summary, location):
        self.link = link # string
        self.followersCount = followersCount # int
        self.friendsCount = friendsCount # int
        self.statusesCount = statusesCount # int
        self.summary = summary # string
        self.location = location # string
    def show(self):
        print "link:", self.link
        print "location:", self.location
        print "followersCount:", self.followersCount
        print "friendsCount:", self.friendsCount
        print "statusesCount:", self.statusesCount
        #print "summary:", self.summary

class EventPair:
    def __init__ (self, string, freq):
        words = string.split()
        self.event1 = ""
        self.event2 = ""
        self.event1_trigger = ""
        self.event2_trigger = ""
        self.relation = ""
        self.freq = freq
        relation_idx = -1
        angle_brackets_count = 0
        event_triggerList = []
        end_idx = -1
        for i, word in enumerate(words):
            if word in set(["Preventative-Effect", "Cause-Effect", "Catalyst-Effect", "Before-After", "MitigatingFactor-Effect", "Precondition-Effect"]):
                self.relation = word
                relation_idx = i
            if word[0] == "[":
                event_triggerList.append(word)
            if word in ["<", ">"]:
                angle_brackets_count += 1
            if angle_brackets_count == 4:
                end_idx = i
                break
        
        self.event1 = " ".join(words[:relation_idx])
        self.event2 = " ".join(words[relation_idx+1:end_idx+1])

        self.event1_trigger = event_triggerList[0]
        self.event2_trigger = event_triggerList[1]

def process_eventpair_knowledge(file):
    eventpairList = []
    with open(file, "r") as input_file:
        for line in input_file:
            if not line.strip():
                continue
            items = line.split(" | ")
            freq = int(items[1].replace("\n", ""))
            if freq <= 1:
                continue
            eventpair = EventPair(items[0], freq)
            eventpairList.append(eventpair)
    return eventpairList

def pairs_to_chains(eventpairList, output_file):
    
    event_set = set()
    G = nx.DiGraph()
    for eventpair in eventpairList:
        edge = (eventpair.event1_trigger, eventpair.event2_trigger)
        event_set.add(eventpair.event1_trigger)
        event_set.add(eventpair.event2_trigger)
        G.add_edge(*edge)
    #nx.draw(G)
    #plt.savefig("path_graph1.png")
    #plt.show()
    output = open(output_file, "w", 0)
    for event1 in event_set:
        for event2 in event_set:
            if event1 == event2:
                continue
            #for path in nx.all_simple_paths(G, source=event1, target = event2):
            #    output.write(str(path) + "\n")
            if nx.has_path(G, source = event1, target = event2):
                path = nx.shortest_path(G, source = event1, target = event2)
                output.write(str(path) + "\n")
    output.close()

def postedTime2seconds(postedTime):
    D = int(postedTime[8:10])
    H = int(postedTime[11:13])
    M = int(postedTime[14:16])
    S = int(postedTime[17:19])
    return (D-20)*24*3600 + H * 3600 + M * 60 + S

def get_timeList():
    dateList = []
    for date in range(22, 32):
        dateList.append("2017_08_" + str(date))
    for date in range(1, 16):
        if date < 10:
            date = "0" + str(date)
        dateList.append("2017_09_" + str(date))

    hourList = []
    for hour in range(0, 24):
        if hour < 10:
            hour = "0" + str(hour)
        hourList.append(str(hour))

    minuteList = []
    for minute in range(0, 6):
        minute = str(minute) + "0"
        minuteList.append(minute)

    return dateList, hourList, minuteList

def HubsAndAuthorities(ID2node, srcID2dstIDs, dstID2srcIDs):
    k_steps = 100
    for i in range(0, k_steps):
        norm = 0
        for ID in ID2node:
            ID2node[ID].auth = 0
            if ID in dstID2srcIDs:
                for srcID in dstID2srcIDs[ID]:
                    ID2node[ID].auth += ID2node[srcID].hub
            norm += ID2node[ID].auth ** 2
        norm = norm ** (0.5)
        for ID in ID2node:
            ID2node[ID].auth = ID2node[ID].auth / norm
        norm = 0
        for ID in ID2node:
            ID2node[ID].hub = 0
            if ID in srcID2dstIDs:
                for dstID in srcID2dstIDs[ID]:
                    ID2node[ID].hub += ID2node[dstID].auth
            norm += ID2node[ID].hub ** 2
        norm = norm ** (0.5)
        for ID in ID2node:
            ID2node[ID].hub = ID2node[ID].hub / norm
    return ID2node

def load_stopwords(file):
    stopwords = set()
    with open(file, "r") as input_file:
        for line in input_file:
            if not line.strip():
                continue
            words = line.split()
            stopwords.add(words[0])
    return stopwords

def load_twitter_stopwords(file):
    stopwords = set()
    with open(file, "r") as input_file:
        for line in input_file:
            if not line.strip():
                continue
            words = line.split(",")
            for word in words:
                stopwords.add(word)
    return stopwords

def load_important_words(file):
    category2important_words = {}
    with open(file, "r") as input_file:
        for line in input_file:
            if not line.strip():
                continue
            category = line.split(" | ")[0]
            important_words = set()
            for word in line.split(" | ")[1].split():
                if word[0] == '%':
                    continue
                elif word[0] == '@':
                    important_words.add(word.replace('@', ''))
                else:
                    important_words.add(word)
            
            category2important_words[category] = important_words
    return category2important_words

def load_event_ontology(file):
    input_file = open(file, "r").readlines()
    phase_category2keywords = {}
    expression2words = {}
    for line in input_file:
        if not line.strip():
            continue
        if line[0] == "@":
            words = []
            expression = line.split()[0]
            phrases = " ".join(line.split()[1:]).split(",")
            for phrase in phrases:
                words.append(" ".join(phrase.split()).lower())
            expression2words[expression] = words
    for line in input_file:
        if not line.strip():
            continue
        words = line.split()
        if words[0] == "#":
            phase = words[1]
            phase_category2keywords[phase] = {}
            continue
        if words[0] == "##":
            category = "_".join(words[1:])
            phase_category2keywords[phase][category] = []
            continue
        if words[0] == "###":
            phrases = line.replace("###", "").split(",")
            for phrase in phrases:
                expression_flag = False
                for word in phrase.split():
                    if "@" + word in expression2words:
                        expression_flag = True
                        for expression_word in expression2words["@" + word]:
                            final_phrase = " ".join(phrase.split()).replace(word, expression_word)
                            phase_category2keywords[phase][category].append(final_phrase.lower())
                if expression_flag == False:
                    phase_category2keywords[phase][category].append(" ".join(phrase.split()).lower())
            continue
    return phase_category2keywords

"""
stop = set(stopwords.words('english'))
stopwords = load_stopwords("../dic/stopwords.txt")
stop = stop | stopwords

stop = set()
exclude = set(string.punctuation)
exclude = exclude - set(["#", "@"])
lemma = WordNetLemmatizer()
p.set_options(p.OPT.URL, p.OPT.EMOJI, p.OPT.MENTION)

def clean(doc):
    doc = doc.replace(u"\u2026", "")
    doc = p.clean(doc)
    stop_free = []
    for word in doc.split():
        if word[0] == "#":
            stop_free.append(word)
            continue
        if word not in stop and len(word) >= 3 and not word.isdigit():
            stop_free.append(word.lower())
    stop_free = " ".join(stop_free)
    punc_free = ''.join(ch for ch in stop_free if ch not in exclude)
    normalized = " ".join(lemma.lemmatize(word) for word in punc_free.split())
    return normalized
"""

def build_dict(docList, V):
    dic = {}
    for doc in docList:
        for word in doc.split():
            if word not in dic:
                dic[word] = 1
            else:
                dic[word] += 1
    sorted_dic = sorted(dic.items(), key = lambda e: e[1], reverse = True)
    word2idx = {}
    idx2word = {}
    count = 0
    for item in sorted_dic:
        count += 1
        idx2word[count] = item[0]
        word2idx[item[0]] = count
        if count > V:
            break
    return word2idx, idx2word

#model = gensim.models.KeyedVectors.load_word2vec_format('../../tools/GoogleNews-vectors-negative300.bin', binary = True)

def doc_vec(doc, word2idx):
    sum_vec = np.zeros(300)
    count = 0
    for word in doc.split():
        if word[0] == "#":
            continue
            """
            print word
            for subword in re.findall('[A-Z][^A-Z]*', word.replace("#", "")):
                print subword
                try:
                    sum_vec += model[subword]
                    count += 1
                except KeyError:
                    pass
            continue
            """
        if word not in word2idx:
            continue
        try:
            sum_vec += model[word]
            count += 1
        except KeyError:
            pass
    if count <= 2:
        return False, None
    else:
        return True, sum_vec / float(count)


def similarity(u, v):
    return metrics.pairwise.cosine_similarity([u], [v])[0][0]

def find_nearest(u, category2centroid):
    min_category = None
    min_distance = None
    for category in category2centroid:
        distance = similarity(u, category2centroid[category])
        if min_distance == None:
            min_category = category
            min_distance = distance
        elif min_distance < distance:
            min_category = category
            min_distance = distance
    return min_category, min_distance


def extract_tweet_info(tweet):
    """
    return json.dumps(tweet, indent = 4, sort_keys=True) + "\n\n"
    """
    new_tweet = {}
    flag = False
    if "long_object" in tweet and "body" in tweet["long_object"]:
        new_tweet["body"] = tweet["long_object"]["body"]
    elif "object" in tweet and "long_object" in tweet["object"] and "body" in tweet["object"]["long_object"]:
        new_tweet["body"] = tweet["object"]["long_object"]["body"]
    elif "object" in tweet and "body" in tweet["object"]:
        new_tweet["body"] = tweet["object"]["body"]
    elif "body" in tweet:
        new_tweet["body"] = tweet["body"]
        flag = True

    # (retweet format) RT @poemless: 
    if "body" in tweet and flag == False:
        words = tweet["body"].split()
        if len(words) >= 2 and words[0] == "RT" and words[1][0] == "@":
            new_tweet["body"] = words[0] + " " + words[1] + " " + new_tweet["body"]
    

    if "actor" in tweet:
        new_tweet["actor_Username"] = tweet["actor"]["preferredUsername"]
        new_tweet["actor_id"] = tweet["actor"]["id"].split(":")[-1]
        new_tweet["actor_followersCount"] = tweet["actor"]["followersCount"]
        new_tweet["actor_link"] = tweet["actor"]["link"]
    if "postedTime" in tweet:
        new_tweet["postedTime"] = tweet["postedTime"]
    if "link" in tweet:
        new_tweet["link"] = tweet["link"]
        words = tweet["link"].split("/")
        new_tweet["tweet_id"] = words[3] + "_" + words[5]
    if "geo" in tweet:
        new_tweet["geo"] = tweet["geo"]["coordinates"]
    if "twitter_lang" in tweet:
        new_tweet["twitter_lang"] = tweet["twitter_lang"]
    return new_tweet

def extract_tweet_info_Florence(tweet):
    """
    return json.dumps(tweet, indent = 4, sort_keys=True) + "\n\n"
    """
    new_tweet = {}
    if "retweeted_status" in tweet and "full_text" in tweet["retweeted_status"] and "text" in tweet:
        words = tweet["text"].split()
        if words[0] == "RT" and words[1][0] == "@":
            new_tweet["body"] = words[0] + " " + words[1] + " " + tweet["retweeted_status"]["full_text"]
    elif "extended_tweet" in tweet and "full_text" in tweet["extended_tweet"]:
        new_tweet["body"] = tweet["extended_tweet"]["full_text"]
    elif "text" in tweet:
        new_tweet["body"] = tweet["text"]

    if "user" in tweet:
        new_tweet["actor_Username"] = tweet["user"]["screen_name"]
        new_tweet["actor_id"] = tweet["user"]["id"]
        new_tweet["actor_followersCount"] = tweet["user"]["followers_count"]
        new_tweet["actor_link"] = ""
        new_tweet["tweet_id"] = new_tweet["actor_Username"] + "_" + tweet["id_str"]
        new_tweet["link"] = "https://twitter.com/" + new_tweet["actor_Username"] + "/statuses/" + tweet["id_str"]
        if "derived" in tweet["user"] and "locations" in tweet:
            new_tweet["user_location"] = tweet["user"]["derived"]["locations"][0]["full_name"]

    if "created_at" in tweet:
        words = tweet["created_at"].split()
        new_tweet["postedTime"] = "2018-09-" + words[2] + "T" + words[3] + words[4].replace("+", ".")
    
    if "geo" in tweet and tweet["geo"] != None:
        new_tweet["geo"] = tweet["geo"]["coordinates"]
    if "lang" in tweet:
        new_tweet["twitter_lang"] = tweet["lang"]

    return new_tweet


# https://pypi.org/project/langdetect/
def English_detector(wordList):
    new_wordList = []
    for word in wordList:
        if word[0].isalpha() and "http" not in word:
            new_wordList.append(word)
    if len(new_wordList) == 0:
        return False
    try:
        if "en" in str(detect_langs(" ".join(new_wordList))[0]):
            return True
        else:
            return False
    except LangDetectException:
        return False


def tweet_filter(tweet, hourList):

    if "body" not in tweet or "chunkList" not in tweet or "link" not in tweet or "postedTime" not in tweet:
        return True
    
    if tweet["body"].split()[0] == "RT":
        return True

    if tweet["body"][0] == "@":
        return True

    if len(tweet["chunkList"]) <= 1:
        return True

    hour = tweet["postedTime"].split("T")[1].split(":")[0]
    if hour not in hourList:
        return True

    return False

# above filter is too strict, will miss many tweets
def test_tweet_filter(tweet, hourList):

    if "body" not in tweet or "link" not in tweet or "postedTime" not in tweet:
        return True
    
    if tweet["body"].split()[0] == "RT":
        return True

    if tweet["body"][0] == "@":
        return True

    s = p.clean(tweet["body"])
    if len(s.split()) <= 2:
        return True

    hour = tweet["postedTime"].split("T")[1].split(":")[0]
    if hour not in hourList:
        return True

    return False

def process_text(string):
    #x = re.sub('[^A-Za-z0-9]+', ' ', x)
    string = string.lower()
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)     
    string = re.sub(r"\'s", " \'s", string) 
    string = re.sub(r"\'ve", " \'ve", string) 
    string = re.sub(r"n\'t", " n\'t", string) 
    string = re.sub(r"\'re", " \'re", string) 
    string = re.sub(r"\'d", " \'d", string) 
    string = re.sub(r"\'ll", " \'ll", string) 
    string = re.sub(r",", " , ", string) 
    string = re.sub(r"!", " ! ", string) 
    string = re.sub(r"\(", " \( ", string) 
    string = re.sub(r"\)", " \) ", string) 
    string = re.sub(r"\?", " \? ", string) 
    string = re.sub(r"\s{2,}", " ", string)    
    #string = string.split(' ')
    #string = [strip_punctuation(s) for s in string]
    # ptxt = nltk.word_tokenize(ptxt)
    return string


if __name__ == "__main__":
    T = "1st 3rd They're opening flood gates to alleviate the pressure on dams &amp; reservoirs. Highways are closed &amp; underwater. Thousands are stranded"
    cleaned_tweet_words = p.clean(T)
    print cleaned_tweet_words # p.clean() processes 1st 3rd wrongly
    cleaned_tweet_words = process_text(cleaned_tweet_words)
    cleaned_tweet_words = set(cleaned_tweet_words.split())
    print cleaned_tweet_words

