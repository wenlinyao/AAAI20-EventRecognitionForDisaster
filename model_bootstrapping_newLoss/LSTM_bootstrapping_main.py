"""
This file is the main fuction that takes arguments, prepare training and testing data, and train the neural network.
"""
import sys
sys.path.append("../utilities/")
reload(sys)
sys.setdefaultencoding('utf-8')
from utilities import load_event_ontology
from LSTM_process_data import create_database_context
from bootstrapping_utils import keywords_tweet_id2categories, SLPA_tweet_id2categories
from bootstrapping_utils import prepare_data, get_eval_tweet_id2categories, update_tweet_id2categories

from multiprocessing import Process
from LSTM_train_context import LSTM_main_context
import random, time, argparse, os, sys, math, pickle, ast


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    #self.parser.add_argument("--rnn_type", dest="rnn_type", type=str, metavar='<str>', default='LSTM', help="Recurrent unit type (lstm|gru|simple) (default=lstm)")
    parser.add_argument("--opt", dest="opt", type=str, metavar='<str>', default='Adam', help="Optimization algorithm (RMS|Adam|SGD|Adagrad|Adadelta) (default=Adam)")
    parser.add_argument("--emb_size", dest="embedding_size", type=int, metavar='<int>', default=300, help="Embeddings dimension (default=300)")
    parser.add_argument("--rnn_size", dest="rnn_size", type=int, metavar='<int>', default=300, help="RNN dimension. '0' means no RNN layer (default=300)")
    parser.add_argument("--batch_size", dest="batch_size", type=int, metavar='<int>', default=100, help="Batch size (default=100)")
    parser.add_argument("--rnn_layers", dest="rnn_layers", type=int, metavar='<int>', default=1, help="Number of RNN layers (default = 1)")
    parser.add_argument("--aggregation", dest="aggregation", type=str, metavar='<str>', default='max', help="The aggregation method for regp and bregp types (mean|last|max) (default=max)")
    parser.add_argument("--dropout", dest="dropout", type=float, metavar='<float>', default=0.5, help="The dropout probability. To disable, give a negative number (default=0.5)")
    #parser.add_argument("--pretrained", dest="pretrained", type=int, metavar='<int>', default=1, help="Whether to use pretrained or not")
    parser.add_argument("--epochs", dest="epochs", type=int, metavar='<int>', default=30, help="Number of epochs (default=30)")
    parser.add_argument("--maxlen", dest="maxlen", type=int, metavar='<int>', default=25, help="Maximum allowed number of words during training.")
    parser.add_argument('--gpu', dest='gpu', type=int, metavar='<int>', default=0, help="Specify which GPU to use (default=0)")
    parser.add_argument("--hdim", dest='hidden_layer_size', type=int, metavar='<int>', default=300, help="Hidden layer size (default=300)")
    parser.add_argument("--lr", dest='learn_rate', type=float, metavar='<float>', default=0.0001, help="Learning Rate (default=0.001)")
    parser.add_argument("--clip", dest='clip', type=float, metavar='<float>', default=5.0, help="Gradient clipping")
    parser.add_argument("--trainable", dest='trainable', type=bool, metavar='<bool>', default=False, help="Trainable Word Embeddings (default=False)")
    parser.add_argument('--l2_reg', dest='l2_reg', type=float, metavar='<float>', default=0.0, help='L2 regularization, default=0')
    parser.add_argument('--eval', dest='eval', type=int, metavar='<int>', default= 5, help='Epoch to evaluate results (default=5)')
    parser.add_argument('--dev', dest='dev', type=int, metavar='<int>', default=1, help='1 for development set 0 to train-all')
    parser.add_argument('--toy', dest='toy', type = bool, metavar='<bool>', default=False, help="Use toy dataset (for fast testing), True means use toy dataset")
    parser.add_argument('--seed', type=int, default=11, help='random seed')
    parser.add_argument('--cuda', dest='cuda', type = bool, metavar='<bool>', default=False, help='use CUDA')
    parser.add_argument('--trained_model', dest = 'trained_model', type = str, metavar='<str>', default=None, help="Use pre-trained model (default=None means no pretrained model)")
    parser.add_argument('--class_num', dest='class_num', type=int, metavar='<int>', default=10, help="Total number of classes (default=7, 6 classes + other)")
    parser.add_argument("--context_size", dest="context_size", type=int, metavar='<int>', default=1, help="Context window size (default=1), update in each bootstrapping iteration")
    parser.add_argument("--neg_sample_r", dest="neg_sample_r", type=float, metavar='<float>', default=0.2, help="Negative sampling rate")
    parser.add_argument("--keywords_drop_r", dest="keywords_drop_r", type=float, metavar='<float>', default=0.3, help="Keywords drop rate")
    parser.add_argument("--confidence_t", dest="confidence_t", type=float, metavar='<float>', default=0.5, help="Classifier confidence threshold (default=0.5)")
    parser.add_argument("--word_flag", dest="word_flag", type=str, metavar='<str>', default="lemma", help="Word form flag in training and testing (default=lemma)")
    parser.add_argument("--data_source", dest="data_source", type=str, metavar='<str>', default="Harvey", help="Harvey or Florence")

    args = parser.parse_args()

    random.seed(args.seed)
    

    
    folder = "../run_SNLI_encoder4/"

    if args.data_source == "Harvey":
        input_name2train_hourList = {"2017_08_28_tweetList.txt": ["06", "07", "08", "09", "10", "11", "12", "13", "14", "15", "16", "17", "18"]}
        
        input_name2test_hourList = {"2017_08_28_tweetList.txt": ["18"]}
        
    elif args.data_source == "Florence":
        # !!!!!!!!!!!!!!!!! Florence !!!!!!!!!!!!!!!!!!!
        input_name2train_hourList = {"Florence_2018_09_17_tweetList.txt": ["06", "07", "08", "09", "10", "11", "12", "13", "14", "15", "16", "17"]}
        input_name2test_hourList = {"Florence_2018_09_17_tweetList.txt": ["18"]}
    

    current_folder = ""


    #                       0                     1                2            3    
    labelList = ["#Preventative_measure", "#Help_and_rescue", "#Casualty", "#Housing", 
    #                       4                        5                         6
                "#Utilities_and_Supplies", "#Transportation", "#Flood_control_infrastructures", 
    #                      7                          8                   9
                "#Business_Work_School", "#Built-environment_hazards", "#Other"]

    phase_category2keywords = load_event_ontology("../dic/event_ontology_new.txt")

    category2important_words = {}
    
    for category in phase_category2keywords["Impact"]:
        category2important_words["#" + category] = set(phase_category2keywords["Impact"][category])

    assert args.class_num == len(labelList)

    
    train_tweet_ids, test_tweet_ids, actor2tweet_ids, tweet_id2tweet = prepare_data(folder, input_name2train_hourList, input_name2test_hourList)
    pickle.dump(test_tweet_ids, open("test_tweet_ids.p", "wb"))
    pickle.dump(train_tweet_ids, open("train_tweet_ids.p", "wb"))
    pickle.dump(actor2tweet_ids, open("actor2tweet_ids.p", "wb"))
    pickle.dump(tweet_id2tweet, open("tweet_id2tweet.p", "wb"))
    
    

    test_tweet_ids = pickle.load(open("test_tweet_ids.p", "rb"))
    train_tweet_ids = pickle.load(open("train_tweet_ids.p", "rb"))

    actor2tweet_ids = pickle.load(open("actor2tweet_ids.p", "rb"))

    # original_tweet_id2reply_ids.p is generated from extract_specific_tweets/extract_reply_tweets.py
    original_tweet_id2reply_ids = pickle.load(open("original_tweet_id2reply_ids.p", "rb"))

    tweet_id2tweet = pickle.load(open("tweet_id2tweet.p", "rb"))

    start_i = 1
    #end_i = 2
    end_i = 15

    args.context_size = 5 # !!!
    #args.context_size = 0
    
    args.neg_sample_r = 0.3
    args.neg_sample_r = 0.4
    args.neg_sample_r = 0.5

    args.keywords_drop_r = 0.2 # !!!
    #args.keywords_drop_r = 0.1
    #args.keywords_drop_r = 0
    #args.keywords_drop_r = 1
    
    #args.confidence_t = 0.5 # !!!
    args.confidence_t = 0.9

    #args.word_flag = "word"
    args.word_flag = "lemma" # !!!

    new_pred_count = 999
    #new_pred_count = 157
    switch_flag = 0

    output = open("performance_log.txt", "w", 0)
    for i in range(start_i, end_i):
        start = time.time()


        
        if new_pred_count < 100:
            if args.confidence_t <= 0.5:
                break
            else:
                args.confidence_t -= 0.1
       

        print "args.keywords_drop_r, args.confidence_t, args.neg_sample_r:", args.keywords_drop_r, args.confidence_t, args.neg_sample_r
        print "process_data_main(...)"

        
        if i - 1 == 0:
            tweet_id2categories = keywords_tweet_id2categories(folder, input_name2train_hourList, category2important_words)
            
            pickle.dump(tweet_id2categories, open("tweet_id2categories_0.p", "wb"))
        
        
        


        last_tweet_id2categories = pickle.load(open("tweet_id2categories_" + str(i-1) + ".p", "rb"))
        eval_tweet_id2categories = get_eval_tweet_id2categories(last_tweet_id2categories, test_tweet_ids)
        print "##################", "tweet_id2categories_" + str(i - 1) + ".p", "#####################"
        
        if args.trained_model == None:
            create_database_context(tweet_id2tweet, labelList, last_tweet_id2categories, actor2tweet_ids, original_tweet_id2reply_ids,
                train_tweet_ids, test_tweet_ids, category2important_words, "../../tools/glove.6B/glove.6B.300d.txt", args)
        
        print "LSTM_main(...)"
        LSTM_main_context(current_folder, args)

        # update tweet_id2categories
        tweet_id2categories, new_pred_count = update_tweet_id2categories("LSTM_true_and_pred_value.txt", labelList, last_tweet_id2categories, args.confidence_t)

        pickle.dump(tweet_id2categories, open("tweet_id2categories_" + str(i) + ".p", "wb"))
        
        eval_tweet_id2categories = get_eval_tweet_id2categories(tweet_id2categories, test_tweet_ids)
        print "##################", "tweet_id2categories_" + str(i) + ".p", "#####################"
        output.write("################## tweet_id2categories_" + str(i) + ".p #####################\n")
        output.write("new_pred_count: " + str(new_pred_count) + "\n")
        output.write("args.keywords_drop_r, args.confidence_t, args.neg_sample_r: " + str(args.keywords_drop_r) + " " + str(args.confidence_t) + " " + str(args.neg_sample_r) + "\n")
        

        end = time.time()

    output.close()
    
