import numpy as np
import pandas as pd
import pickle

train_df = pd.read_csv('/Users/ducvu/Downloads/dataset/dataset-6.csv')

print(len(train_df[train_df['label'] == 1]))
print(len(train_df[train_df['label'] == 0]))

train_texts = train_df['body'].tolist()
train_texts = [str(char).lower() for char in train_texts]

import os
os.environ["GOOGLE_APPLICATION_CREDENTIALS"]='/Users/ducvu/Downloads/apikey.json'

from cleantext  import clean
from google.cloud import translate_v2 as translate

import contractions
from tqdm import tqdm
import time

translate_client = translate.Client()

cleaned_posts = []
try: 
	for i in tqdm(range(5601, 5769)):
		post = train_texts[i]
		    
		post = contractions.fix(post)
		  
		post = clean(post, 
		             fix_unicode = True, to_ascii = True, 
		             lower = True, no_line_breaks = True, no_urls = True, 
		             replace_with_url = ' ',
		             no_punct = True,
		             lang = 'en')
		  
		translation = translate_client.translate(post, target_language='vi')
		  
		trans_post = translation['translatedText']

		cleaned_posts.append(trans_post)

		time.sleep(1)

except Exception as e:
	with open('dataset4.pk', 'wb') as f:
		pickle.dump(cleaned_posts, f) 


