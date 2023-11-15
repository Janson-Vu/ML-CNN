# -*- coding: utf-8 -*-
"""Updated model.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1DNpmuT6D0fF7NouUhJ97MBjqX49iJKir

> IMPORTING PACKAGES FOR PREPROCESSING
"""

import numpy as np
import pandas as pd

train_df = pd.read_csv('/content/dataset.csv')
train_df.head()

train_texts = train_df['body'].tolist()
train_texts = [ str(char).lower() for char in train_texts]

train_texts[1]

train_texts[-1]

train_texts

print(len(train_df[train_df['label'] == 1]))
print(len(train_df[train_df['label'] == 0]))

!pip install clean-text
!pip install contractions
!pip install textsearch
!pip install tqdm

from cleantext  import clean
from google.cloud import translate
import contractions
from tqdm import tqdm


cleaned_posts = []
for i in tqdm(range(len(train_texts))):
  
  post = train_texts[i]
    
  post = contractions.fix(post)
  
  post = clean(post, 
               fix_unicode = True, to_ascii = True, 
               lower = True, no_line_breaks = True, no_urls = True, 
               replace_with_url = ' ',
               no_punct = True,
               lang = 'en')
  
  
  cleaned_posts.append(post)

cleaned_posts

print(len(cleaned_posts))

"""> Convert to string index"""

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

# Tokenizer
t = Tokenizer(num_words=None, char_level=True, oov_token='UNK')
t.fit_on_texts(train_texts)

alphabet="abcdefghijklmnopqrstuvwxyz0123456789 ,;.!?:'\"/\\|_@#$%^&*~`+-=<>()[]{}"
char_dict = {}
for i, char in enumerate(alphabet):
    char_dict[char] = i + 1
    
# Use char_dict to replace the tk.word_index
t.word_index = char_dict.copy() 
# Add 'UNK' to the vocabulary 
t.word_index[t.oov_token] = max(char_dict.values()) + 1

# Convert string to index 
train_sequences = t.texts_to_sequences(train_texts)

# Padding
train_data = pad_sequences(train_sequences, maxlen=1014, padding='post')

# Convert to numpy array
train_data = np.array(train_data, dtype='float32')

train_classes = train_df['label'].values
train_class_list = train_classes

from keras.utils import to_categorical
train_classes = to_categorical(train_class_list)

print(train_classes.shape)

print(train_classes[0])

print(train_data.shape)

"""> CONSTRUCT MODEL"""

print(t.word_index)

vocab_size = len(t.word_index)
vocab_size

embedding_weights = [] #(71, 70)
embedding_weights.append(np.zeros(vocab_size)) # first row is pad

for char, i in t.word_index.items(): # from index 1 to 70
    onehot = np.zeros(vocab_size)
    onehot[i-1] = 1
    embedding_weights.append(onehot)
embedding_weights = np.array(embedding_weights)

print(embedding_weights.shape) # first row all 0 for PAD, 69 char, last row for UNK
embedding_weights

from keras.layers import Input, Embedding, Activation, Flatten, Dense
from keras.layers import Conv1D, MaxPooling1D, Dropout
from keras.models import Model

# parameter 
input_size = 1014
vocab_size = 69
embedding_size = 70
conv_layers = [[256, 7, 3], 
               [256, 7, 3], 
               [256, 3, -1], 
               [256, 3, -1], 
               [256, 3, -1], 
               [256, 3, 3]]

fully_connected_layers = [1024, 1024]
num_of_classes = 2
dropout_p = 0.5
optimizer = 'adam'
loss = 'categorical_crossentropy'

# Embedding layer Initialization
embedding_layer = Embedding(vocab_size+2, 
                            embedding_size,
                            input_length=input_size,
                            weights=[embedding_weights])

# Model 

# Input
inputs = Input(shape=(input_size,), name='input', dtype='int64')  # shape=(?, 1014)

# Embedding 
x = embedding_layer(inputs)

# Conv 
for filter_num, filter_size, pooling_size in conv_layers:
    x = Conv1D(filter_num, filter_size)(x) 
    x = Activation('relu')(x)
    if pooling_size != -1:
        x = MaxPooling1D(pool_size=pooling_size)(x) # Final shape=(None, 34, 256)
        
x = Flatten()(x) # (None, 8704)

# Fully connected layers 
for dense_size in fully_connected_layers:
    x = Dense(dense_size, activation='relu')(x) # dense_size == 1024
    x = Dropout(dropout_p)(x)
    
# Output Layer
predictions = Dense(num_of_classes, activation='softmax')(x)

# Build model
model = Model(inputs=inputs, outputs=predictions)
model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy']) # Adam, categorical_crossentropy
#model.summary()

"""> TRAIN MODEL"""

#np.random.shuffle(train_data)

train_data = train_data
train_class_list = train_class_list[:]
print(train_data.shape)
print(train_class_list.shape)

len(train_data)/2

x_train = train_data
y_train = train_classes

# Training
model.fit(x_train, y_train,
          validation_split = 0.4,
          batch_size=128,
          epochs=10, 
          verbose = 2)

text1 = 'I couldn’t pull myself together this morning. Again. I woke up, went to the bathroom to shower and start my day, and just fell to the floor in tears. I couldn’t even turn on the water. I tried so hard to pull it together. I tried every coping mechanism possible: deep breathing, mindfulness, rationalization. You name it. I’m ruining my life because my brain is sick. I see a therapist and I’m medicated. I don’t know what I’m doing wrong.' 
text2 = "It's actually a quite close relative to the North American House Hippo. That's the American Bantam Bison. If you ever stop in Yellowstone you should check your vehicle for these guys, while small in stature they still maintain a similar weight to the full sized ones. Kind of like Ant Man in specific scenes where they remembered they said he did that."

def predict(text):
  text = contractions.fix(text)

  text = clean(text, 
               fix_unicode = True, to_ascii = True, 
               lower = True, no_line_breaks = True, no_urls = True, 
               replace_with_url = ' ',
               no_punct = True,
               lang = 'en')

  instance = t.texts_to_sequences(text)

  flat_list = []
  for sublist in instance:
    for item in sublist:
      flat_list.append(item)

  flat_list = [flat_list]

  instance = pad_sequences(flat_list, padding='post', maxlen=1014)

  prediction = model.predict(instance)
  
  return prediction

print(predict(text1)) # first one is non-depressed, second one is depressed