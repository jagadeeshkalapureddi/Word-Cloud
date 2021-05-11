# -*- coding: utf-8 -*-
"""
Created on Fri Feb 12 14:12:11 2021

@author: Jagadeesh K
"""
#############################################################################    
'Word Clouding'
#############################################################################
import matplotlib.pyplot as pPlot
from wordcloud import WordCloud, STOPWORDS
import numpy as npy
from PIL import Image

dataset = open("Acknowledgment.txt", "r").read()
def create_word_cloud(string):
   maskArray = npy.array(Image.open("cloud.png"))
   cloud = WordCloud(background_color = "white", max_words = 500, 
                     mask = maskArray, stopwords = set(STOPWORDS))
   cloud.generate(string)
   cloud.to_file("wordCloud.png")
   pPlot.imshow(cloud, interpolation = 'bilinear')
   pPlot.show
dataset = dataset.lower()
create_word_cloud(dataset)

#############################################################################
'Text Mining' '- Vinod Sir' '- Achnowledgement.txt'
#############################################################################
#import pandas as pd
#import numpy as np
import matplotlib.pyplot as plt
#import seaborn
import re
#import string
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
#from nltk.stem import PorterStemmer
#from nltk.stem import WordNetLemmatizer
import nltk
#from wordcloud import wordcloud

with open('Acknowledgment.txt', 'r') as f:
    text = f.readlines()

text = [line.strip() for line in text]

text

# Let us have a vector of words and apply paste function into it

a = ["Hello", "World","My", "First","Handshake", "Through", "R"]
a

# We are looking to use paste function and make it a chunk

b = ' '.join(a)
b

# Just chect how to do these two look

a
b

# Lets start again

with open('Acknowledgment.txt', 'r') as f:
    chunk_2 = f.readlines()

chunk_2 = [line.strip() for line in text]
print(chunk_2)

chunk_pasted_2 = ' '.join(chunk_2)
chunk_pasted_2


## Let us now lower case this data

clean_data1 = chunk_pasted_2.lower()
clean_data1


## Cleaning the punctuations

clean_data2 = re.sub(r'[^\w\s]','',clean_data1)
clean_data2

## Digits.

clean_data3 = re.sub(r'\d+', ' ', clean_data2)
clean_data3

## Stop Words
nltk.download('stopwords')

# Lets 
stop_words = set(stopwords.words('english'))
stop_words

# Let us remove them using function removewords()
from nltk.tokenize import word_tokenize
import nltk

nltk.download('punkt')

tokens = word_tokenize(clean_data3)

clean_data4 = [i for i in tokens if not i in stop_words]

clean_data4

# lets club the list

clean_data4 = " ".join(str(x) for x in clean_data4)
clean_data4

## let us remove single letters

clean_data5 = ' '.join(i for i in clean_data4.split() if not (i.isalpha() and 
                                                              len(i)==1))
clean_data5

## Whitespace
clean_data6 = clean_data5.strip()
clean_data6

## Frequency of the words

words_dict = {}
for word in clean_data6.split():
    words_dict[word] = words_dict.get(word, 0)+1
for key in sorted(words_dict):
    print("{}:{}".format(key,words_dict[key]))

## Word Cloud

# Create the wordcloud object
wordcloud1 = WordCloud(width=480, height=480, margin=0).generate(clean_data6)

plt.figure()
plt.imshow(wordcloud1, interpolation='bilinear')
plt.axis("off")
plt.show()

# Create the wordcloud object
wordcloud1 = WordCloud(width=480, height=480, max_words=3).generate(clean_data6)

plt.imshow(wordcloud1, interpolation='bilinear')
plt.axis("off")
plt.show()

## Sentiment Analysis

from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk

nltk.download('vader_lexicon')

analyser = SentimentIntensityAnalyzer()

scores = analyser.polarity_scores(clean_data6)

for key in sorted(scores):
    print('{0}: {1}, '.format(key, scores[key]), end='')

#############################################################################

#############################################################################
'Text Mining' '- Vinod Sir' '- Data Science Article.txt'
#############################################################################
#import pandas as pd
#import numpy as np
import matplotlib.pyplot as plt
#import seaborn
import re
#import string
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
#from nltk.stem import PorterStemmer
#from nltk.stem import WordNetLemmatizer
import nltk
#from wordcloud import wordcloud

with open('DSa.txt', 'r') as f:
    text = f.readlines()
 
text = [line.strip() for line in text]

text

# Let us have a vector of words and apply paste function into it

a = ["Hello", "World","My", "First","Handshake", "Through", "R"]
a

# We are looking to use paste function and make it a chunk

b = ' '.join(a)
b

# Just chect how to do these two look

a
b

# Lets start again

with open('DSa.txt', 'r') as f:
    chunk_2 = f.readlines()

chunk_2 = [line.strip() for line in text]
print(chunk_2)

chunk_pasted_2 = ' '.join(chunk_2)
chunk_pasted_2

## Let us now lower case this data

clean_data1 = chunk_pasted_2.lower()
clean_data1


## Cleaning the punctuations

clean_data2 = re.sub(r'[^\w\s]','',clean_data1)
clean_data2

## Digits.

clean_data3 = re.sub(r'\d+', ' ', clean_data2)
clean_data3

## Stop Words
nltk.download('stopwords')

# Lets 
stop_words = set(stopwords.words('english'))
stop_words

# Let us remove them using function removewords()
from nltk.tokenize import word_tokenize
import nltk

nltk.download('punkt')

tokens = word_tokenize(clean_data3)

clean_data4 = [i for i in tokens if not i in stop_words]

clean_data4

# lets club the list

clean_data4 = " ".join(str(x) for x in clean_data4)
clean_data4

## let us remove single letters

clean_data5 = ' '.join(i for i in clean_data4.split() if not (i.isalpha() and 
                                                              len(i)==1))
clean_data5

## Whitespace
clean_data6 = clean_data5.strip()
clean_data6

## Frequency of the words

words_dict = {}
for word in clean_data6.split():
    words_dict[word] = words_dict.get(word, 0)+1
for key in sorted(words_dict):
    print("{}:{}".format(key,words_dict[key]))

## Word Cloud

# Create the wordcloud object
maskArray = npy.array(Image.open("gg.png"))
wordcloud1 = WordCloud(width=580, height=600, margin=0, mask = maskArray).generate(clean_data6)

plt.figure()
plt.imshow(wordcloud1)
plt.axis("off")
plt.show()

#############################################################################

