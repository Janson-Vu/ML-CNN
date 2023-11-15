import tensorflow as tf

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

import pandas as pd
import numpy as np

import contractions
from cleantext import clean

train_df = pd.read_csv('/Users/ducvu/Downloads/ISEF/dataset/dataset-6.csv')
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


model = tf.keras.models.load_model('/Users/ducvu/Downloads/ISEF/ISEF_1st_models/english_model.h5')

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


d_text1 = "No matter how I try to make myself forget about the way I feel it always resurfaces and I feel alone. I cried at school all day and people kept asking if I’m okay . I said yes because no matter how I try to explain I feel no one will truly understand. It’s not like I haven’t tried to explain but my feelings either get invalidated or tossed to the side. I want to be happy but my environment is restricting me. I always thought “I can’t be depressed when people have it worse than me” Today I realized that’s incorrect. For the past two days I’ve asked God to take me off this Earth like I did two months ago . Here I am. Alone."
d_text2 = "Why it is that other people makes it look easy to have success in doing things while me constantly failing in everything I do.Like when I tried doing Shopify (selling goods online) many people are having so much success profiting thousands of dollars while me failing and losing money running ads.When I tried to learn coding other students can easily get the logic and solve the problem while me getting stuck in a problem and can’t think of any solution for it.I also tried blogging, cpa, ptc and many other online money making methods but failed.I prefer to work online because I have halitosis for 13+ years now and it is very difficult for me to communicate or mingle with other people while having super stinky breath (I have a separate reddit thread about my halitosis) but working online is not working for me.I think my brain is too weak for this kinds jobs.Every day waking up I always feel that I’m the yuckiest and dumbest person in the world because apart from having bad breathe I’m also failing in everything I do.I can’t find any skill that that I can use to earn money.I’m very useless!"
d_text3 = "They never listen to you. But the feeling of pulling them out is actually quite comfortable. Removing those hairs feels like you're getting rid of yourself. They are deep, broken hair. Just like yourself. You continue to pluck the hair until the skin on the top of your head like a forest has been completely exploited, completely. So you still can't stop there? Just stop your hand, do not put your hand on the head alone can not do. You just spit and eventually switch to your sparse scalp until it bleeds. Blood flows from your head to your forehead and you're still acting like a madman? But actually that spit and trust has brought you peace of mind even though you are destroying yourself. Your scalp becomes sore, stroking your hair can help you feel an area of ​​hair shortage. You have low self-esteem but now you have made yourself ugly. Is there anything worse?"
n_text2 = "I really wanted an Audi S4 wagon, but we don't get those in the US. So like any rational person, I got a SQ5 and tried to make it into a wagon. I ended up pushing things kinda far and now have a really stupid fast CUV. It doesn't make sense. It shouldn't exist. It does though.Started with a 2015 SQ5 and added a few mods.Performance: 034 beta E40 ECU & TCU, upgraded pulleys (3.276 PR) on a ported supercharger, 75mm tb, ams heat exchanger with separate reservoir, intake, upgraded HPFP, Handling: KW V3 coilovers, 034 rear sway bar, x-brace, solid drivetrain mounts, spherical control arms, HRE P40SC wrapped in Toyo R888.Styling: I wash it regularly."
d_text4 = "But you really want to get out of that reality, everything is too tiring and boring. Excuse me for dreaming so loudly, apologize to myself for vaguely choosing to pick up the other knife to end my life."
d_text5 = "Even when I'm really hungry, I really don't want anything, I just want to sleep deeply, forget about hunger. You once again accidentally harmed yourself. How ironic. Just a small cut makes you cry"
#print(predict(d_text3)) # first one is non-depressed, second one is depressed
n_text = 'i am very happy'

print(d_text1)
print(predict(d_text1))
