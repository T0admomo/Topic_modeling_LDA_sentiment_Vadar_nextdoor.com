#!/usr/bin/env python
# coding: utf-8

# ## Read and Write Json File

# In[13]:


import pickle 
import json
import glob
import gensim
import pandas as pd
from gensim.utils import simple_preprocess
from spacy.tokens import DocBin
import pandas as pd
from sklearn.feature_extraction import text
import spacy
from gensim.models import Word2Vec


# In[ ]:


def load_json(file):
    with open (file, 'r',encoding = 'utf-8') as f:
        data = json.load(f)
    return data

def write_json(file, data):
    with open (file, 'w',encoding = 'utf-8') as f:
        json.dump(data, f, indent = 4)

def load_pkl(file):
    with open(file,'rb') as raw_pickle:
         data = pickle.load(raw_pickle)
    return data

def write_pkl(file, data):
    with open(file,'wb') as raw_pickle:
         pickle.dump(data, raw_pickle) 
            
            
            
def load_data(version):   
    """
    Takes a file name and 
    returns token_texts, corpus, id2word
    """
    
    token_text = load_json(f'../data/tokens/{version}.json')

    corpus = load_json(f'../data/corpi/{version}.json')

    id2word = load_pkl(f'../data/word_ids/{version}.pkl')
    
    return token_text, corpus, id2word


# # Tf-IDF REMOVAL

# ___

# We arn't going to want to move forward with this kind of word removal until we have done more manual inspection and stop word removal. The brevity of our texts may demand that we keep as many words as possible, and lossing frequently occurring words seminal to the focus of the study .i.e homelessness would prevent us from identifying a posts relevancy overall at this time. for example. we still do not currently know if all of our posts pertain to homelessness, or if they include many discussions of cats up tree's. Untill we have done more cleaning, we will want our topic model to to isolate irrelavant posts, removing "homeless" from all of the posts would make it rather difficult to do this. So it is a step better saved for latter.

# In[23]:


# from gensim.models import TfidfModel

# # create word dictionary 
# id2word = corpora.Dictionary(data_bigrams_trigrams)
# # just to make it simpler going forward 
# texts = data_bigrams_trigrams
# # convert all of our texts into a bag of words
# corpus = [id2word.doc2bow(text) for text in texts]

# print ( corpus[0][0:20])

# # instantiate tfidf model 
# tfidf = TfidfModel(corpus, id2word=id2word)

# low_value = 0.03
# words = []
# words_missing_in_tfidf = []


# for i in range(0, len(corpus)):
#     bow = corpus[i]
#     low_value_words = [] 
#     tfidf_ids = [id for id, value in tfidf[bow]]
#     bow_ids = [id for id, value in bow]
#     low_value_words = [id for id, value in tfidf[bow] if value < low_value] 
#     drops = low_value_words+words_missing_in_tfidf
#     for item in drops:
#         words.append(id2word[item])
#     words_missing_in_tfidf = [id for id in bow_ids if id not in tfidf_ids] # the words with tf-idf score will be missing
    
#     new_bow= [b for b in bow if b[0] not in low_value_words and b[0] not in words_missing_in_tfidf]
#     corpus[i] = new_bow
    
# corpus


# ## Word Vectors

# In[2]:




def sim_word(kwds, top_n ):
    """
    kwds : list of search terms
    top_n: number of similar term to return 
    
    returns: top_n most similar words for each of the terms in kwds
    """
    
    for kwd in kwds:
        try:
            print(kwd)
            res = model.wv.similar_by_word(kwd, topn=top_n)
            for term in res:
                print(term)
        except KeyError:
            print('[]"Word not in vocabulary"[]]')
            continue


# ## Lemma

# In[3]:




def lemmatization(texts, allowed_postages=['NOUN','ADJ','VERB','ADV']):
    
    '''
    allowed_postage : the parts of speach we want to keep [DEFAULT: 'NOUN','ADJ','VERB','ADV'] 
    '''
    
          # load in spacy sm web model 
    nlp = spacy.load('en_core_web_lg', diasble = ['parser','ner']) # computaltionally expensize aspects 
    texts_out = [] # output
    
    # for each post in the corpus
    # iterate over texts
    for text in texts:  
        # creates spacy doc object containing vectorized contextual information like Parts of Speech (pos) 
        doc = nlp(text)
                
        # list for holding lemmatized tokens
        new_text = []
        # iterate over each token
        for token in doc:
            # only keep the desired pos
            if token.pos_ in allowed_postages:
                # cleans minimal stop words and punctuation 
                if not token.is_stop == True:
                    # reducing model complexity by reducing tokens to lemma_ 
                    new_text.append(token.lemma_)   
                    # print(token.lemma_)

        final = " ".join(new_text)
        texts_out.append(final)
    return (texts_out)


def gen_words(texts):
    final = [] 
    for text in texts:
        # Convert a document into a list of lowercase tokens, ignoring tokens that are too short or too long.
        new = gensim.utils.simple_preprocess(text , deacc=True)# – Remove accent marks from tokens using 
        final.append(new)
    return(final)


# ## Word Grams

# In[4]:


def make_bigrams(texts):
    return [bigram[doc] for doc in texts]

def make_trigrams(texts):
    return [trigram[bigram[doc]] for doc in texts]
# https://spacy.io/usage/training#quickstart


# In[5]:


def gen_words(texts):
    final = [] 
    for text in texts:
        # new = mwe.tokenize(text)
        # Convert a document into a list of lowercase tokens, ignoring tokens that are too short or too long.
        new = gensim.utils.simple_preprocess(text , deacc=True)# – Remove accent marks from tokens using 
        final.append(new)
    return(final)


# # Bigrams &  Trigrams 

# ___

# We attempt to capture some of the more important word pairings with bigrams and trigrams. 

# In[22]:


# https://www.youtube.com/watch?v=UEn3xHNBXJU
# https://www.machinelearningplus.com/nlp/topic-modeling-gensim-python/#9createbigramandtrigrammodels

## set parameters

# def make_bigrams(texts):
#     return [bigram[doc] for doc in texts]

# def make_trigrams(texts):
#     return [trigram[bigram[doc]] for doc in texts]


# # Build the bigram and trigram models
# bigram_model = gensim.models.Phrases(data_words, min_count = 3, threshold = 50 )# min freq for a coupling to be a bigram ## thresh = num of bigrams allowes
# # of the bigrams, are is their overlap in the rest of our words for a trigram?
# trigram_model = gensim.models.Phrases(bigram_model[data_words], threshold = 50 )

## create 

# # Faster way to get a sentence clubbed as a trigram/bigram

# # fit bigram model 
# bigram = gensim.models.phrases.Phraser(bigram_model)
# trigram = gensim.models.phrases.Phraser(trigram_model)

# # instantia
# data_bigrams = make_bigrams(data_words)
# data_bigrams_trigrams = make_trigrams(data_bigrams)

# print(data_bigrams_trigrams)


# # Tuning 

# In[9]:


# supporting function
from gensim.models import CoherenceModel 

def compute_coherence_values(corpus, text_tokens, id2word, k, a, b, c ):
    
    """
    corpus: text body in string form
    dictionary: id2word 
    k: num_topics
    a: alpha - document topic density
    b: beta- word topic density 
    
    """
    
    lda_model = gensim.models.ldamodel.LdaModel(corpus=corpus,
                                           id2word=id2word,
                                           num_topics=k, 
                                           random_state=100,
                                           chunksize=c,
                                           passes=10,
                                           alpha=a,
                                           eta=b)
    
    coherence_model_lda = CoherenceModel(model=lda_model, texts=text_tokens, dictionary=id2word, coherence='c_v')
    
    return coherence_model_lda.get_coherence()


# In[10]:


def tune_lda(corpus, text_tokens, id2word, t_range, c_range, a_range, b_range ):
    # Can take a long time to run
    # result df entry format
    model_results = {'Topics': [],
                     'Alpha': [],
                     'Beta': [],
                     'Chunks': [],
                     'Coherence': []
                    }
    # iterate through number of topics
    for k in t_range:
        # iterate through alpha values
        for a in a_range:
              # iterare through beta values
            for b in b_range:
                # iterate through chunksizes
                for c in c_range:
                     # get cohenerence value

                    cv = compute_coherence_values( corpus, text_tokens, id2word, 
                                                  k, a, b, c = c)
                    # Save the model results
                    model_results['Topics'].append(k)
                    model_results['Alpha'].append(a)
                    model_results['Beta'].append(b)
                    model_results['Chunks'].append(c)
                    model_results['Coherence'].append(cv)
    return pd.DataFrame(model_results)


# In[ ]:


def gen_similarity(word, filepath):
    model = KeyedVectors.load_word2vec_format(f'{filepath}', binary = False)
    results = model.most_similar(positive =[word])
    print(results)


# In[20]:


def sim_word(kwds, top_n):
    """
    kwds : list of search terms
    top_n: number of similar term to return 
    
    returns: top_n most similar words for each of the terms in kwds
    """
    # for each search term 
    for kwd in kwds:
        try:
            print(kwd.upper()) # show search term 
            res = model.wv.similar_by_word(kwd, topn=top_n) # find similar words
            for term in res:
                print(term) # show similar words
        except KeyError:
            print('[]"Word not in vocabulary"[]]')
            continue
    


# ## For creating a labeled dataset

# In[24]:


def cat_posts(post_tokens, kwds, label):    
    '''
    label: class label 
    kwds: search terms
    post_tokens: tokenized corpus
    
    returns: labeled dataset containing target posts
    '''
    
    train_data = []   
    match = None
    # if a post has one of our kwds
    for word in kwds:
        for post in post_tokens:
            match = False 
            # only relevant posts 
            if word in post:
                match = True
            # if our keyword is in a post then match = True    
            # attaches given class label to post
            if match == True:
                post = ' '.join(post)
                train_data.append((post,label))

    return train_data


# In[11]:


import json 
from gensim.models.word2vec import Word2Vec
from gensim.models.keyedvectors import KeyedVectors
import multiprocessing  # for using seperate core's


# In[12]:


def train_w2v(model_name, filepath):
    with open (f'{filepath}', 'r', encoding = 'utf-8') as f:
        texts = json.load(f)
    sentence = texts 
    cores = multiprocessing.cpu_count()
    w2v_model = Word2Vec(
            min_count = 2,
            window = 2, # num of surrounding words to consider 
            # shape of a word vector
            size = 500,  #size of vocab , dimensions of vocab                  
            sample = 6e-5,
            alpha =0.03, # error term
            min_alpha=0.0007, # ??
            negative = 20, # ??
            workers = cores-1 # number of cores to train with  
        )
    
    w2v_model.build_vocab(texts) # create the model vocabulary 
    w2v_model.train(texts, total_examples=w2v_model.corpus_count, epochs = 30) # train the model
    w2v_model.save(f'../data/word_vec_model_{model_name}.model') # save the model for comparison 
    w2v_model.wv.save_word2vec_format(f'../data/word_vectors_{model_name}.txt') # save word vectors
    # https://www.youtube.com/watch?v=1  

