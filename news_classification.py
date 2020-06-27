#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun 27 07:27:46 2020

@author: zxs
"""

# Import libraries
import pandas as pd
import os
import time
from gensim import corpora
import numpy as np
import matplotlib.pyplot as plt
from gensim.models import LdaMulticore
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from multiprocessing import Pool, cpu_count
from sklearn.model_selection import train_test_split
from sklearn import metrics
from joblib import dump

%matplotlib inline
%config InlineBackend.figure_format = 'retina'

plt.style.use('fivethirtyeight')

os.chdir('/Users/zxs/Documents/code/')

from zfuncs import tag_dat, process_dat
from sklearn.ensemble import RandomForestClassifier
from wordcloud import WordCloud

# Read the data
os.chdir('/Users/zxs/Documents/data')

real = pd.read_csv('news/real_news.csv')
fake = pd.read_csv('news/fake_news.csv')

# Add labels
real['label'] = 1
fake['label'] = 0

# Combine sets
print('Reading Data..')
data = pd.concat([fake, real])
print('Processing text..')

# Instatntiate multiprocessing
cpus = cpu_count() - 1
p = Pool(cpus)

# Inspect
print('Topic distribution of documents by label:')
print(data.groupby('subject')['label'].value_counts())
print('Processing time:')

# Apply function to text in parallel
%time processed_text = p.map(process_dat, data['text'])
p.close()

data['processed_text'] = processed_text

data['processed_title'] = data['title'].apply(lambda x: process_dat(x))

# Further process results
data['proc_txt'] = data['processed_text'].apply(lambda x: ', '.join(x))
data['proc_ttl'] = data['processed_title'].apply(lambda x: ', '.join(x))

data['txt'] = data['proc_ttl'] + ', ' + data['proc_txt']
data['txt'] = data['txt'].apply(lambda x: x.replace(', ', ' ').strip())

# Create a dictionary for the corpus and give each unique term an index
dictionary = corpora.Dictionary(data['txt'].apply(lambda x: x.split(' ')))

# Filter infrequent tokens
dictionary.filter_extremes(no_below = 100, no_above = 0.8)

# Create a term document matrix (corpus)
term_mat = [dictionary.doc2bow(comment) for comment in data['txt'].apply(lambda x: x.split(', '))]

# Vectorize text
v = CountVectorizer(analyzer = 'word', min_df = 100, max_features = 100)
d = v.fit_transform(data['txt'])

print('Building Models..')

# Split data for training
y = data['label']
x = d

xtrain, xval, ytrain, yval = train_test_split(x, y, random_state = 100, test_size = .2)
 
# Baseline classification
rfc = RandomForestClassifier(random_state = 100)
rfc.fit(xtrain, ytrain)
rfc_preds = rfc.predict(xval)

print('Baseline RFC Performance: ')
print(metrics.confusion_matrix(yval, rfc_preds))
print(metrics.classification_report(yval, rfc_preds))

# ROC curve
fpr, tpr, _ = metrics.roc_curve(yval, rfc_preds)
roc_auc = metrics.auc(fpr, tpr)
    
plt.figure(figsize = [25, 10])
plt.plot(fpr, tpr, color = 'darkorange', label = 'ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC for RF Model of News Data')
plt.legend(loc="lower right")
plt.show()

# Save model for use elsewhere
dump(rfc, 'fakenews_rfc.joblib') 
