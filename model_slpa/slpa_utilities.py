import sys
reload(sys)
sys.setdefaultencoding('utf-8')
import random
import numpy as np
from collections import defaultdict
import preprocessor as p
import ast, pickle, time, os, math, copy
from multiprocessing import Process
from numpy import dot
from numpy.linalg import norm

class Graph():
    def __init__(self):
        self.nodes = set()
        self.edges = {}

    def add_edge(self, u, v, weight=1.0):
        assert u != v
        if u not in self.nodes:
            self.nodes.add(u)
        if v not in self.nodes:
            self.nodes.add(v)
        if u not in self.edges:
            self.edges[u] = {}
            self.edges[u][v] = weight
        else:
            if v in self.edges[u]:
                self.edges[u][v] += weight
            else:
                self.edges[u][v] = weight
    def has_edge(self, u, v):
        if u in self.edges and v in self.edges[u]:
            return True
        else:
            return False

class MetaGraph():
    def __init__(self):
        self.nodes = set()
        self.edges = {}

    def add_edge(self, u, v, weight):
        assert u != v
        if u not in self.nodes:
            self.nodes.add(u)
        if v not in self.nodes:
            self.nodes.add(v)
        if u not in self.edges:
            self.edges[u] = {}
            self.edges[u][v] = [weight]
        else:
            if v in self.edges[u]:
                self.edges[u][v] += [weight]
            else:
                self.edges[u][v] = [weight]
    def has_edge(self, u, v):
        if u in self.edges and v in self.edges[u]:
            return True
        else:
            return False

def MetaGraph2Graph(MetaG, type2w):
    G = Graph()
    w_types = type2w.keys()
    G.nodes = MetaG.nodes
    for u in MetaG.edges:
        for v in MetaG.edges[u]:
            for w_type in w_types:
                count = MetaG.edges[u][v].count(w_type)
                if count == 0:
                    continue
                
                if w_type == "I_G_w": 
                    #if count == 1:
                    #    continue
                    w = 2.0 ** (count - 2.0) * type2w["I_G_w"]
                    G.add_edge(u, v, w)
                    G.add_edge(v, u, w)

                elif w_type == "W_G_w":
                    if count == 1:
                        continue
                    w = 2.0 ** (count - 2.0) * type2w["W_G_w"]
                    G.add_edge(u, v, w)
                    G.add_edge(v, u, w)

                elif w_type == "G_w":
                    if count == 1:
                        continue
                    w = 2.0 ** (count - 2.0) * type2w["G_w"]
                    G.add_edge(u, v, w)
                    G.add_edge(v, u, w)

                #else:
                #    G.add_edge(u, v, float(count) * type2w[w_type])
                #    G.add_edge(v, u, float(count) * type2w[w_type])
    return G

def cos_sim(a, b):
    return dot(a, b)/(norm(a)*norm(b))


def get_acceptedLabel(G, i, memory, listenersOrder, memory2distribution, label2norm_w):

    listener2acceptedLabel = {}
    for j, listener in enumerate(listenersOrder):
        if j % 20 != i:
            continue
        if listener not in G.edges:
            continue
        speakers = G.edges[listener].keys()
        
        labels = defaultdict(int)
        num_speakers = len(speakers)
        for j, speaker in enumerate(speakers):
            if speaker not in memory2distribution:
                continue
            # Speaker Rule
            #total = float(sum(memory[speaker].values()))
            #distribution = [freq/total for freq in memory[speaker].values()]
            distribution = memory2distribution[speaker]
            chosen_idx = np.random.multinomial(1, distribution).argmax()
            chosen_label = memory[speaker].keys()[chosen_idx]
            #print speaker, memory[speaker], distribution, chosen_idx
            labels[chosen_label] += 1.0 * G.edges[listener][speaker] # * label2norm_w[chosen_label]

        # Listener Rule
        if len(labels) == 0:
            continue
        acceptedLabel = max(labels, key=labels.get)
        listener2acceptedLabel[listener] = acceptedLabel
    pickle.dump(listener2acceptedLabel, open("listener2acceptedLabel_" + str(i) + ".p", "wb"))

def initialize_memory(G, seed_tweet_idx2labels, type_flag):
    ##Stage 1: Initialization
    print "initialize memory", type_flag
    initial_weight = 60

    if type_flag == "1": # each node is a cluster
        memory = {i:{i:1} for i in list(G.nodes)}

    elif type_flag == "2": # only use seed labels
        memory = {}
        count = 0
        for i in list(G.nodes):
            memory[i] = {}
            if i in seed_tweet_idx2labels:
                count += 1
                labels = seed_tweet_idx2labels[i]
                for label in labels:
                    memory[i][label] = initial_weight / len(labels)
                    #memory[i][label] = 1
            else:
                memory[i][i] = 1
        print "nodes having initial label:", count, "total nodes:", len(G.nodes), "ratio:", float(count) / float(len(G.nodes))
    
    elif type_flag == "3": # use seed labels and the cluster result from pass 1
        memory = {}
        communities = pickle.load(open("communities_pass1.p", "rb"))
        for key in communities:
            for idx in communities[key]:
                memory[idx] = {key:1}

        countList = []
        for i in list(G.nodes):
            if i in seed_tweet_idx2labels:
                memory[i] = {}
                countList.append("A")
                labels = seed_tweet_idx2labels[i]
                for label in labels:
                    memory[i][label] = initial_weight / len(labels)
            elif i in memory:
                countList.append("B")
                pass
            else:
                countList.append("C")
                memory[i] = {i:1}

        print "nodes having initial labels:", countList.count("A"), "total nodes:", len(G.nodes), "ratio:", float(countList.count("A")) / float(len(G.nodes))
        print "nodes having communities pass 1 labels:", countList.count("B"), "total nodes:", len(G.nodes), "ratio:", float(countList.count("B")) / float(len(G.nodes))
        print "nodes having no labels:", countList.count("C"), "total nodes:", len(G.nodes), "ratio:", float(countList.count("C")) / float(len(G.nodes))

    return memory
    

def find_communities(G, T, r, decay_r, memory):
    """
    Speaker-Listener Label Propagation Algorithm (SLPA)
    see http://arxiv.org/abs/1109.5720
    https://github.com/romain-fontugne/slpa_nx
    """
    communitiesList = []
    memoryList = []
    ##Stage 2: Evolution
    for t in range(T):
        start = time.time()
        print "epoch", t,
        listenersOrder = list(G.nodes)
        np.random.shuffle(listenersOrder)
        memory2distribution = {}

        for node in memory:
            total = float(sum(memory[node].values()))
            memory2distribution[node] = [freq/total for freq in memory[node].values()]

        label2total_num = {}
        for node in memory:
            for label in memory[node]:
                if label not in label2total_num:
                    label2total_num[label] = float(memory[node][label])
                else:
                    label2total_num[label] += float(memory[node][label])
        avg_num = float(sum(label2total_num.values())) / float(len(label2total_num))

        label2norm_w = {}
        for label in label2total_num:
            label2norm_w[label] = avg_num / float(label2total_num[label])



        processV = []
        for i in range(0, 20):
            processV.append(Process(target = get_acceptedLabel, args = (G, i, memory, listenersOrder, memory2distribution, label2norm_w, )))
        for i in range(0, 20):
            processV[i].start()
        for i in range(0, 20):
            processV[i].join()

        for i in range(0, 20):
            listener2acceptedLabel = pickle.load(open("listener2acceptedLabel_" + str(i) + ".p", "rb"))
            # Update listener memory
            for listener in listener2acceptedLabel:
                acceptedLabel = listener2acceptedLabel[listener]
                if acceptedLabel in memory[listener]:
                    memory[listener][acceptedLabel] += 1.0 * decay_r ** t
                else:
                    memory[listener][acceptedLabel] = 1.0 * decay_r ** t
            os.system("rm " + "listener2acceptedLabel_" + str(i) + ".p")

        end = time.time()
        print "    time:", end - start
        communitiesList.append(extract_communities(T, r, memory))
        memoryList.append(copy.deepcopy(memory))
    pickle.dump(communitiesList, open("communitiesList.p", "wb"))
    pickle.dump(memoryList, open("memoryList.p", "wb"))
    return extract_communities(T, r, memory)

def find_communities_test(G_test, nodes_test, T, r, existing_memory):
    communitiesList = []
    memoryList = []
    ##Stage 1: Initialization
    memory = {i:{i:1} for i in nodes_test}
    memory.update(existing_memory)

    ##Stage 2: Evolution
    for t in range(T):
        start = time.time()
        print "epoch", t
        listenersOrder = nodes_test
        np.random.shuffle(listenersOrder)
        memory2distribution = {}

        for node in memory:
            total = float(sum(memory[node].values()))
            memory2distribution[node] = [freq/total for freq in memory[node].values()]

        processV = []
        for i in range(0, 20):
            processV.append(Process(target = get_acceptedLabel, args = (G_test, i, memory, listenersOrder, memory2distribution, )))
        for i in range(0, 20):
            processV[i].start()
        for i in range(0, 20):
            processV[i].join()

        for i in range(0, 20):
            listener2acceptedLabel = pickle.load(open("listener2acceptedLabel_" + str(i) + ".p", "rb"))
            # Update listener memory
            for listener in listener2acceptedLabel:
                acceptedLabel = listener2acceptedLabel[listener]
                if acceptedLabel in memory[listener]:
                    memory[listener][acceptedLabel] += 1
                else:
                    memory[listener][acceptedLabel] = 1
            os.system("rm " + "listener2acceptedLabel_" + str(i) + ".p")

        end = time.time()
        print "    time:", end - start
        communitiesList.append(extract_communities(T, r, memory))
        memoryList.append(copy.deepcopy(memory))
    pickle.dump(communitiesList, open("test_communitiesList.p", "wb"))
    pickle.dump(memoryList, open("test_memoryList.p", "wb"))
    return extract_communities(T, r, memory)


def extract_communities(T, r, memory):
    memory = copy.deepcopy(memory)
    ## Stage 3:
    for node, mem in memory.iteritems():
        w_sum = 0
        for label, freq in mem.items():
            w_sum += freq
        for label, freq in mem.items():
            if freq/w_sum < r:
                del mem[label]


    # Find nodes membership
    communities = {}
    for node, mem in memory.iteritems():
        for label in mem.keys():
            if label in communities:
                communities[label].add(node)
            else:
                communities[label] = set([node])


    # Remove nested communities
    nestedCommunities = set()
    keys = communities.keys()
    for i, label0 in enumerate(keys[:-1]):
        comm0 = communities[label0]
        for label1 in keys[i+1:]:
            comm1 = communities[label1]
            if comm0.issubset(comm1):
                nestedCommunities.add(label0)
            elif comm0.issuperset(comm1):
                nestedCommunities.add(label1)
    
    for comm in nestedCommunities:
        del communities[comm]

    return communities

def graph_density(G):
    edge_count = 0
    """
    for node1 in G.nodes:
        for node2 in G.nodes:
            if node1 == node2:
                continue
            if G.has_edge(node1, node2) == True:
                edge_count += 1
    """
    for node1 in G.edges:
        for node2 in G.edges[node1]:
            if node1 == node2:
                continue
            edge_count += 1

    if len(G.nodes) == 0:
        return 0, 0, 0
    else:
        return float(edge_count) / float(len(G.nodes) ** 2 - len(G.nodes)), edge_count, len(G.nodes)
