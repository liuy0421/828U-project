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
from sklearn.preprocessing import normalize

from tensor_lda.tensor_lda import TensorLDA

import numpy as np
import pandas as pd
import pickle
import scipy.sparse as sparse

print("Loading data...")

# t0=time()
file_path = "DATA/GSM3828672_Smartseq2_GBM_IDHwt_processed_TPM.tsv"
df = pd.read_csv(file_path, sep='\t').transpose()
# print("done in %0.3fs." % (time() - t0))

# print("Filtering genes...")
# t0=time()

# extract gene names
gene_names = df.values[0]

# normalize expression value by cell, so all cells are in the same level.
exp_cell_normalized = np.round(normalize(df.values[1:],axis=1,norm='max')*100)

# Keep the top 2000 genes with the highest variance
var = exp_cell_normalized.var(axis = 0)

num_genes_list = [500,1000,1500]

num_topics = [3,5,7,9,11]

for n_genes in num_genes_list:
    ind = np.argpartition(var, -n_genes)[-n_genes:]

    # print("done in %0.3fs." % (time() - t0))

    # print("Constructing sparse matrix...")
    # t0=time()

    tf = sparse.csr_matrix(exp_cell_normalized[:,ind].astype(float))
    tf_feature_names = gene_names[ind]

    with open("results/filtered_gene%d_exp.pkl"%(n_genes), 'wb') as f:  # Python 3: open(..., 'wb')
        pickle.dump([tf,tf_feature_names], f,-1)

    # print("done in %0.3fs." % (time() - t0))

    n_samples = tf.shape[0]
    n_features = tf.shape[1]
    n_top_words = 30

    for n_components in num_topics:

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


        # print("Doing tensor LDA...")
        # t0=time()

        lda = TensorLDA(n_components=n_components, alpha0=.01)

        # t0 = time()
        lda.fit(tf)
        # print("done in %0.3fs." % (time() - t0))

        prediction = lda.transform(tf).argmax(axis=1)
        print("\n")

        print("# of topic: %d\t # of gene: %d" % (n_components, n_genes))

        print("Topics in LDA model:")
        print_top_words(lda, tf_feature_names, n_top_words)
        print("\n\n")


        with open("results/lda_model_topic%d_gene%d.pkl"%(n_components, n_genes), 'wb') as f:  # Python 3: open(..., 'wb')
            pickle.dump(lda, f,-1)

# Getting back the objects:
# with open('objs.pkl') as f:  # Python 3: open(..., 'rb')
#     obj0, obj1, obj2 = pickle.load(f)
