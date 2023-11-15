import warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
import nltk
import matplotlib.pyplot as plt
import contractions
import ftfy
import re

from math import exp
from numpy import sign

from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from gensim.models import KeyedVectors
from nltk.corpus import stopwords
from nltk import PorterStemmer

from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Conv1D, MaxPooling1D, LSTM
from tensorflow.keras.layers import Input, Embedding, Activation, Flatten, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

## CONSTANTS

np.random.seed(1234)

DEPRESS_NROWS = 3200  # number of rows to read from the csv
RANDOM_NROWS = 12000  # number of rows to read from the csv

MAX_SEQ_LENGTH = 140  # maximum tweet size
MAX_NUMBER_WORDS = 20000
EMBEDDING_DIM = 300

TRAIN_SPLIT = 0.6
TEST_SPLIT = 0.2
LEARNING_RATE = 0.1
EPOCHS = 10

## SECTION 1: LOAD DATA
DEPRESSION_CSV = '/Users/ducvu/Desktop/DetectDepressionInTwitterPosts-master/depressive_tweets_processed.csv'
RANDOM_CSV = '/Users/ducvu/Desktop/DetectDepressionInTwitterPosts-master/SentimentAnalysisDataset2.csv'
EMBEDDING_FILE = '/Users/ducvu/Desktop/DetectDepressionInTwitterPosts-master/GoogleNews-vectors-negative300.bin.gz'

depressive_df = pd.read_csv(DEPRESSION_CSV, sep = '|', header = None, usecols = range(0,9), nrows = DEPRESS_NROWS)
random_df = pd.read_csv(RANDOM_CSV, encoding = 'ISO-8859-1', usecols = range(0,4), nrows = RANDOM_NROWS)


## SECTION 2: PROCESS DATA
# load pretrained Word2Vec model
# get the embedding of any word using ".word_vec(word)"
# get all words using ".vocab"

word2vec = KeyedVectors.load_word2vec_format(EMBEDDING_FILE, binary = True)

# preprocess crude tweets

def expand_contractions(text):
	return contractions.fix(text)

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

# apply clean tweets to depressive_df tweets and random_df
depression_arr = [tweet for tweet in depressive_df[5]]
random_arr = [tweet for tweet in random_df['SentimentText']]

x_d = clean_tweets(depression_arr)
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



# Assigning labels to the depressive tweets and random tweets data
labels_d = np.array([1] * DEPRESS_NROWS)
labels_r = np.array([0] * RANDOM_NROWS)

# Splitting the arrays into test (60%), validation (20%), and train data (20%)
perm_d = np.random.permutation(len(data_d))

idx_train_d = perm_d[:int(len(data_d)*(TRAIN_SPLIT))]
idx_test_d = perm_d[int(len(data_d)*(TRAIN_SPLIT)):int(len(data_d)*(TRAIN_SPLIT+TEST_SPLIT))]
idx_val_d = perm_d[int(len(data_d)*(TRAIN_SPLIT+TEST_SPLIT)):]


perm_r = np.random.permutation(len(data_r))

idx_train_r = perm_r[:int(len(data_r)*(TRAIN_SPLIT))]
idx_test_r = perm_r[int(len(data_r)*(TRAIN_SPLIT)):int(len(data_r)*(TRAIN_SPLIT+TEST_SPLIT))]
idx_val_r = perm_r[int(len(data_r)*(TRAIN_SPLIT+TEST_SPLIT)):]

# Combine depressive tweets and random tweets arrays
data_train = np.concatenate((data_d[idx_train_d], data_r[idx_train_r]))
labels_train = np.concatenate((labels_d[idx_train_d], labels_r[idx_train_r]))

data_test = np.concatenate((data_d[idx_test_d], data_r[idx_test_r]))
labels_test = np.concatenate((labels_d[idx_test_d], labels_r[idx_test_r]))

data_val = np.concatenate((data_d[idx_val_d], data_r[idx_val_r]))
labels_val = np.concatenate((labels_d[idx_val_d], labels_r[idx_val_r]))

# Shuffling
perm_train = np.random.permutation(len(data_train))
data_train = data_train[perm_train]
labels_train = labels_train[perm_train]

perm_test = np.random.permutation(len(data_test))
data_test = data_test[perm_test]
labels_test = labels_test[perm_test]

perm_val = np.random.permutation(len(data_val))
data_val = data_val[perm_val]
labels_val = labels_val[perm_val]



## Build model (LSTM + CNN)

model = Sequential()

# embedded layer
model.add(Embedding(len(embedding_matrix), EMBEDDING_DIM, weights = [embedding_matrix], input_length = MAX_SEQ_LENGTH, trainable = False))

# CNN
model.add(Conv1D(filters = 32, kernel_size = 3, padding = 'same', activation = 'relu'))
model.add(MaxPooling1D(pool_size = 2))
model.add(Dropout(0.2))

# LSTM
model.add(LSTM(300))
model.add(Dropout(0.2))
model.add(Dense(1, activation = 'sigmoid'))


# compile model
model.compile(loss='binary_crossentropy', optimizer='nadam', metrics=['acc'])

early_stop = EarlyStopping(monitor='val_loss', patience=3)

model.fit(data_train, labels_train, 
        validation_data=(data_val, labels_val), 
        epochs=EPOCHS, batch_size=40, shuffle=True,
        callbacks=[early_stop])


#labels_pred = model.predict(data_test)
#labels_pred = np.round(labels_pred.flatten())
#accuracy = accuracy_score(labels_test, labels_pred)
#print("Accuracy: %.2f%%" % (accuracy*100))


model.save('SentimentAnalysis.h5')






















