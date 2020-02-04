Welcome! This is the implementation of the paper "Weakly-supervised Fine-grained Event Recognition on Social Media Texts for Disaster Management" (AAAI 2020).
The project is only for research purpose. If you think our work is useful to you, please also cite our paper. Thanks!


##############################################################################################################
############################################ Preprocessing ###################################################
##############################################################################################################
Please also download stanford-corenlp toolkit and Glove word embedding and put them into directory: "../../tools/stanford-corenlp-full-2017-06-09" and "../../tools/glove.6B/glove.6B.300d.txt"
In this step, we will use pretrained SNLI sentence encoder to get important words for each tweet message.
The pretrained sentence encoder (file "infersent.allnli.pickle") used here is from "https://github.com/facebookresearch/InferSent".
Please download "infersent.allnli.pickle" and put it into SNLI_encoder/encoder/infersent.allnli.pickle
$ mkdir run_SNLI_encoder4/
$ cd run_SNLI_encoder4/
$ python ../SNLI_encoder/SNLI_encoder_main.py


##############################################################################################################
############################################ SLPA clustering #################################################
##############################################################################################################
In this step, we will modify the original SLPA and use the important words generated from the previous stage to support text clustering. This part of code can also 
be applied to other text clustering problems with no need to predefine the number of clusters. You can also try other measures to score the similarity between two sentences (remember to use one threshold to remove less weighted edges). 
Please refer to the paper "SLPA: Uncovering Overlapping Communities in Social Networks via A Speaker-listener Interaction Dynamic Process" (https://arxiv.org/pdf/1109.5720.pdf) for more details of SLPA algorithm.
#1 
$ mkdir run_model_slpa_Word_v4/
$ cd run_model_slpa_Word_v4/
$ python ../model_slpa/slpa_main_Word_v4.py

output: several pickle files, 9 category folders

#2
cd run_model_slpa_Word_v4/
python ../model_slpa/extract_communitiesList_categories.py
output: in each folder of 9 categories, generate clustering results for all slpa iterations: communities_0/ ... communities_N/


#3
cd run_extract_communities_chunks_categories/
python ../model_slpa/extract_communities_chunks_categories.py

output: for each category, generate the summary of each cluster (top words and terms of each cluster)

#4 (human feedback)
Read the summary files in "run_extract_communities_chunks_categories/" and then write all good cluster IDs into human_feedback.txt (have provided one example file). This file will be used as input for the next multi-channel LSTM classifier training.


##############################################################################################################
############################################## LSTM training #################################################
##############################################################################################################
$ mkdir run_model_bootstrapping_0827-0828/
$ cd run_model_bootstrapping_0827-0828/
create human_feedback_exclude.txt
create human_feedback_max20.txt # remember to put "@ ../run_model_slpa_Word_v4_0827-0828/" (where stores all clusters) at the beginning of the file

modify LSTM_bootstrapping_main.py to support 0827-0828
copy original_tweet_id2reply_ids.p (is generated from extract_specific_tweets/extract_reply_tweets.py) to support replies

$ python ../model_bootstrapping_newLoss/LSTM_bootstrapping_main.py --data_source Harvey --cuda True




