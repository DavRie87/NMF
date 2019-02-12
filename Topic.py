# Author: David Riemeier

# Code is a replication of the work from
# Olivier Grisel, Lars Buitinck, Chyi-Kwei Yau
# https://scikit-learn.org/0.18/auto_examples/applications/
# topics_extraction_with_nmf_lda.html

from __future__ import print_function
from time import time

# import Pandas, numpy and scipy for data structures. Use sklearn for NMF

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import NMF
from sklearn.datasets import fetch_20newsgroups

# set sample properties

n_samples = 18000
n_features = 1500
n_topics = 10
n_top_words = 8

def print_top_words(model, feature_names, n_top_words):
    for topic_idx, topic in enumerate(model.components_):
        print("Topic #%d:" % topic_idx)
        print(" ".join([feature_names[i]
                        for i in topic.argsort()[:-n_top_words - 1:-1]]))
        print()

# load 20 newsgroups dataset and vectorize it
# headers, footers and quotes are excluded from dataset
# print current status / compile time
        
t0 = time()
dataset = fetch_20newsgroups(shuffle=True, random_state=1,
                             remove=('headers', 'footers', 'quotes'))
data_samples = dataset.data[:n_samples]
print("done in %0.3fs." % (time() - t0))

# use tf-idf vectorizer for NMF.
# term frequency - inverse document frequency

tfidf_vectorizer = TfidfVectorizer(max_df=0.95, min_df=2,
                                   max_features=n_features,
                                   stop_words='english')
t0 = time()
tfidf = tfidf_vectorizer.fit_transform(data_samples)
print("done in %0.3fs." % (time() - t0))

# fit the NMF model
# run now NMF function

print("Fitting the NMF model with tf-idf features, "
      "n_samples=%d and n_features=%d..."
      % (n_samples, n_features))
t0 = time()
nmf = NMF(n_components=n_topics, random_state=1,
          alpha=.1, l1_ratio=.5).fit(tfidf)
print("done in %0.3fs." % (time() - t0))

# give the final output for topics

print("\nTopics in NMF model:")
tfidf_feature_names = tfidf_vectorizer.get_feature_names()
print_top_words(nmf, tfidf_feature_names, n_top_words)