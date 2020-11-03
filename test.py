"""
================================
Topic extraction with Tensor LDA
================================

This example is modified from scikit-learn's "Topic extraction with
Non-negative Matrix Factorization and Latent Dirichlet Allocation"
example.

This example applies :class:`tensor_lda.tensor_lda.TensorLDA`
on the 20 news group dataset and the output is a list of topics, each
represented as a list of terms (weights are not shown).


"""

from __future__ import print_function
from time import time

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.datasets import fetch_20newsgroups

from tensor_lda.tensor_lda import TensorLDA

import numpy as np
import pandas as pd
import pickle
import scipy.sparse as sparse


file_path = "DATA/GSM3828672_Smartseq2_GBM_IDHwt_processed_TPM.tsv"
df = pd.read_csv(file_path, sep='\t').transpose()

expr_gene = []
for idx, col in df.iteritems():
    exp = (col !=0).sum()/23686
    if exp > 0.1:
        expr_gene.append(idx)

df = df.iloc[:,expr_gene]

tf = sparse.csr_matrix(df.values[1:].astype(float))

n_samples = tf.shape[0]
n_features = tf.shape[1]
n_components = 6
n_top_words = 20




def print_top_words(model, feature_names, n_top_words):
    for topic_idx, topic in enumerate(model.components_):
        topic_prior = model.alpha_[topic_idx]
        message = "Topic #%d (prior: %.3f): " % (topic_idx, topic_prior)
        message += " ".join([feature_names[i]
                             for i in topic.argsort()[:-n_top_words - 1:-1]])
        print(message)
    print()


# Load the 20 newsgroups dataset and vectorize it. We use a few heuristics
# to filter out useless terms early on: the posts are stripped of headers,
# footers and quoted replies, and common English words, words occurring in
# only one document or in at least 95% of the documents are removed.

lda = TensorLDA(n_components=n_components, alpha0=.01)

t0 = time()
lda.fit(tf)
print("done in %0.3fs." % (time() - t0))

print("\nTopics in LDA model:")
tf_feature_names = tf_vectorizer.get_feature_names()
print_top_words(lda, df.values[1], n_top_words)

doc_topics = lda.transform(tf[0:2, :])
print(doc_topics[0, :])
print(data_samples[0])
