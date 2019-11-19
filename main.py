from itertools import chain

import sys
import nltk
import sklearn
import scipy.stats
from sklearn.metrics import make_scorer
from sklearn.model_selection import train_test_split
from sklearn.model_selection import RandomizedSearchCV

import sklearn_crfsuite
from sklearn_crfsuite import scorers
from sklearn_crfsuite import metrics

# matplotlib inline
import matplotlib.pyplot as plt
plt.style.use('ggplot')

nltk.download('conll2002')
print(nltk.corpus.conll2002.fileids())
print(nltk.corpus.conll2002.iob_sents('esp.testa'))