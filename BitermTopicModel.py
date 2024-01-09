import bitermplus as btm
import numpy as np
import pandas as pd
import csv
import pprint
import itertools
import tmplot as tmp

# IMPORTING DATA
df = pd.read_csv(
    'remove_pre_misokin_original.txt', header=None, names=['texts'])
texts = df['texts'].str.strip().tolist()

# PREPROCESSING
# Obtaining terms frequency in a sparse matrix and corpus vocabulary
X, vocabulary, vocab_dict = btm.get_words_freqs(texts)
tf = np.array(X.sum(axis=0)).ravel()
# Vectorizing documents
docs_vec = btm.get_vectorized_docs(texts, vocabulary)
docs_lens = list(map(len, docs_vec))

# Generating biterms
biterms = btm.get_biterms(docs_vec)

# INITIALIZING AND RUNNING MODEL
model = btm.BTM(
    X, vocabulary, seed=3, T=5, M=10, alpha=0.92, beta=0.14)
model.fit(biterms, iterations=20)
p_zd = model.transform(docs_vec)


# METRICS
perplexity = btm.perplexity(model.matrix_topics_words_, p_zd, X, T=5)
coherence = btm.coherence(model.matrix_topics_words_, X, M=10)
# or
#perplexity = model.perplexity_
#coherence = model.coherence_


# LABELS
model.labels_
# or
btm.get_docs_top_topic(texts, model.matrix_docs_topics_)

# 各トピックの上位10単語を出力する
l = []
n_top_words = 10    #出力単語数
for i, topic_dist in enumerate(model.matrix_topics_words_):
    top_words_indices = np.argsort(topic_dist)[::-1][:n_top_words]
    top_words = np.array(vocabulary)[top_words_indices]
    l.append(','.join(top_words))
    print(f"Topic {i}: {' '.join(top_words)}")

print('perplexity:',perplexity)
print('coherence:',coherence)
print(X)
#print(*vocab_dict)
#print(*vocabulary)
'''
#50単語リスト
new_l = []
for i in l:
    new_l.append(i.split(','))
print(list(itertools.chain.from_iterable(new_l)))
'''

