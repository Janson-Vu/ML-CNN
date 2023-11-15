import numpy as np
from flask import Flask, request, jsonify, render_template

import pickle
import tensorflow as tf

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

import pandas as pd
import numpy as np

import contractions
from cleantext import clean


#### LOAING DATASET AND CNN MODEL
app = Flask(__name__, template_folder = './template')

model = tf.keras.models.load_model('/Users/ducvu/Documents/ISEF demo/english_model.h5')
train_df = pd.read_csv('/Users/ducvu/Documents/ISEF demo/dataset-6.csv')
train_texts = train_df['body'].tolist()
train_texts = [ str(char).lower() for char in train_texts]


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


@app.route('/')
def home():
    return render_template('index.html')

# PROCESS PREDICTION TEXT
@app.route('/predict', methods=['POST'])
def predict():

    # get text
    if request.method == 'POST':
        text = request.form['predict_text']

    # # processing raw rext
    # text = contractions.fix(text)
    #
    # text = clean(text,
    #             fix_unicode = True, to_ascii = True,
    #             lower = True, no_line_breaks = True, no_urls = True,
    #             replace_with_url = ' ',
    #             no_punct = True,
    #             lang = 'en')
    #
    # instance = t.texts_to_sequences(text)
    #
    # flat_list = []
    # for sublist in instance:
    #     for item in sublist:
    #         flat_list.append(item)
    #
    # flat_list = [flat_list]
    #
    # instance = pad_sequences(flat_list, padding='post', maxlen=1014)
    #
    # # prediction
    # prediction = model.predict(instance)

    return '''<html>
    <head>
    </head>
    <body>
        <h1>Hello, ''' text '''!</h1>
    </body>
</html>'''
    #render_template('index.html', prediction_text1 = text)



'''@app.route('/results',methods=['POST'])
def results():

    data = request.get_json(force=True)
    prediction = model.predict([np.array(list(data.values()))])

    output = prediction[0]
    return jsonify(output)'''



if __name__ == "__main__":
    app.run(debug=True)
