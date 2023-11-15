import numpy as np
import pandas as pd
import re
import ftfy
import contractions

from gensim.models import KeyedVectors

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk import PorterStemmer

# Load Data & word2vec embedding file
depress_csv = '/Users/ducvu/Desktop/DetectDepressionInTwitterPosts-master/depressive_tweets_processed.csv'
random_csv = '/Users/ducvu/Desktop/DetectDepressionInTwitterPosts-master/SentimentAnalysisDataset2.csv'
embedding_file = '/Users/ducvu/Desktop/DetectDepressionInTwitterPosts-master/GoogleNews-vectors-negative300.bin.gz'

depress_df = pd.read_csv(depress_csv, sep='|', header=None, usecols=range(0,9), nrows = 3200)
random_df = pd.read_csv(random_csv, encoding = 'ISO-8859-1', usecols=range(0,4), nrows = 20000)
word2vec = KeyedVectors.load_word2vec_format(embedding_file, binary=True)

# create array of tweet
depress_arr = [tweet for tweet in depress_df[5]]
random_arr = [tweet for tweet in random_df['SentimentText']]


def clean_tweets(tweets):
    cleaned_tweets = []
    for tweet in tweets:
        tweet = str(tweet)
        # if url links then dont append to avoid news articles
        # also check tweet length, save those > 10 (length of word "depression")
        if re.match("(\w+:\/\/\S+)", tweet) == None and len(tweet) > 10:
            #remove hashtag, @mention, emoji and image URLs
            tweet = ' '.join(re.sub("(@[A-Za-z0-9]+)|(\#[A-Za-z0-9]+)|(<Emoji:.*>)|(pic\.twitter\.com\/.*)", " ", tweet).split())
            
            #fix weirdly encoded texts
            tweet = ftfy.fix_text(tweet)
            
            #expand contraction
            tweet = expand_contractions(tweet)

            #remove punctuation
            tweet = ' '.join(re.sub("([^0-9A-Za-z \t])", " ", tweet).split())

            #stop words
            stop_words = set(stopwords.words('english'))
            word_tokens = nltk.word_tokenize(tweet) 
            filtered_sentence = [w for w in word_tokens if not w in stop_words]
            tweet = ' '.join(filtered_sentence)

            #stemming words
            tweet = PorterStemmer().stem(tweet)
            
            cleaned_tweets.append(tweet)

    return cleaned_tweets



x_d = clean_tweets(depress_arr)
x_r = clean_tweets(random_arr)

# tokenizer
tokenizer = Tokenizer(num_words = MAX_NUMBER_WORDS)
tokenizer.fit_on_texts(x_d + x_r)

# apply tokenizer to depression_arr and random_arr
sequence_d = tokenizer.texts_to_sequences(x_d)
sequence_r = tokenizer.texts_to_sequences(x_r)

# number of words in tokenizer must be <= 20 000
word_index = tokenizer.word_index

# pad sequences to the same length of 140 words
data_d = pad_sequences(sequence_d, maxlen = MAX_SEQ_LENGTH)
data_r = pad_sequences(sequence_r, maxlen = MAX_SEQ_LENGTH)

# Embedding matrix
number_words = min(MAX_NUMBER_WORDS, len(word_index))

embedding_matrix = np.zeros((number_words, EMBEDDING_DIM))

for (word, index) in word_index.items():
	if word in word2vec.vocab and index < MAX_NUMBER_WORDS:
		embedding_matrix[index] = word2vec.word_vec(word)

print(embedding_matrix)





















