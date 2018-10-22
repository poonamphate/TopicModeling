import json

import re
import numpy as np
import pandas as pd
from pprint import pprint

# Gensim
import gensim
import gensim.corpora as corpora
from gensim.utils import simple_preprocess

# spacy for lemmatization
import spacy

# Plotting tools
#import pyLDAvis
#import pyLDAvis.gensim  # don't skip this
#import matplotlib.pyplot as plt 

# Enable logging for gensim - optional
import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.ERROR)

import warnings
warnings.filterwarnings("ignore",category=DeprecationWarning)

# NLTK Stop words
from nltk.corpus import stopwords
stop_words = stopwords.words('english')
stop_words.extend(['from', 'subject', 're', 'edu', 'use'])

def readText(fileName):
    """function to read Json file and print its content"""
    data = []
    with open(fileName) as f:
        for line in f:
            source = json.loads(line)
            data.append(source['text'])
    #print (data)
    #print ("data is", repr(data))
    return data

def preprocessData(data):
    #remove digits
    data = [re.sub("\d+", "", sent) for sent in data]
    
    # Remove new line characters
    data = [re.sub('\s+', ' ', sent) for sent in data]

    # Remove distracting single quotes
    data = [re.sub("\'", "", sent) for sent in data]

    #print ("data after preprocessing is", data)
    return data

def sent_to_words(sentences):
    for sentence in sentences:
        yield(gensim.utils.simple_preprocess(str(sentence), deacc=True))  # deacc=True removes punctuations

# Define functions for stopwords, bigrams, trigrams and lemmatization
def remove_stopwords(texts):
    return [[word for word in simple_preprocess(str(doc)) if word not in stop_words] for doc in texts]

def make_bigrams(texts, bigram_mod):
    return [bigram_mod[doc] for doc in texts]

def make_trigrams(texts, trigram_mod, bigram_mod):
    return [trigram_mod[bigram_mod[doc]] for doc in texts]

def lemmatization(nlp, texts, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV']):
    """https://spacy.io/api/annotation"""
    texts_out = []
    for sent in texts:
        doc = nlp(" ".join(sent)) 
        texts_out.append([token.lemma_ for token in doc if token.pos_ in allowed_postags])
    return texts_out

def main():                      # Define the main function
    print ("main")
    data = readText('topic_modeling_data.json')
    d = preprocessData(data)
    
    #cleaning2(data)
    #print ("data after cleaning: ", repr(data))
        
    data_words = list(sent_to_words(d))
    #print("dara words: ", data_words)
    
    # Build the bigram and trigram models
    bigram = gensim.models.Phrases(data_words, min_count=5, threshold=100) # higher threshold fewer phrases.
    #trigram = gensim.models.Phrases(bigram[data_words], threshold=100)  

    # Faster way to get a sentence clubbed as a trigram/bigram
    bigram_mod = gensim.models.phrases.Phraser(bigram)
    #trigram_mod = gensim.models.phrases.Phraser(trigram)

    # See trigram example
    #print("trigram_mod :", trigram_mod[bigram_mod[data_words[0]]])
    
    # Remove Stop Words
    data_words_nostops = remove_stopwords(data_words)
    #print("data_words_nostops = ", data_words_nostops)
    
    # Form Bigrams
    data_words_bigrams = make_bigrams(data_words_nostops, bigram_mod)
    
    # Initialize spacy 'en' model, keeping only tagger component (for efficiency)
    # python3 -m spacy download en
    nlp = spacy.load('en', disable=['parser', 'ner'])
    
    # Do lemmatization keeping only noun, adj, vb, adv
    data_lemmatized = lemmatization(nlp, data_words_bigrams, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV'])
    #print("data_lemmatized : ", data_lemmatized[:1])
    
    # Create Dictionary
    id2word = corpora.Dictionary(data_lemmatized)
    
    # Create Corpus
    texts = data_lemmatized
    
    # Term Document Frequency
    corpus = [id2word.doc2bow(text) for text in texts]
    
    # View
    #print("corpus = ", corpus[:1])

    # Build LDA model
    lda_model = gensim.models.ldamodel.LdaModel(corpus=corpus,
                                           id2word=id2word,
                                           num_topics=5, 
                                           random_state=100,
                                           update_every=1,
                                           chunksize=100,
                                           passes=5,
                                           alpha='auto',
                                           per_word_topics=True)
    
    # Print the Keyword in the 10 topics
    #pprint(lda_model.print_topics())
    pprint(lda_model.show_topics(num_topics=10, num_words=5, log=False, formatted=True))

if __name__ == '__main__':
    main()
