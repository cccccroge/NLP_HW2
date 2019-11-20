from itertools import chain
import sys

# dataset
import nltk

# model
import gensim
from gensim.models import Word2Vec
from gensim.models import KeyedVectors

# machine learning
import sklearn
import scipy.stats
from sklearn.metrics import make_scorer
from sklearn.model_selection import train_test_split
from sklearn.model_selection import RandomizedSearchCV
import sklearn_crfsuite
from sklearn_crfsuite import scorers
from sklearn_crfsuite import metrics

# plot
import matplotlib.pyplot as plt
plt.style.use('ggplot')



def word2features(sent, i):
    word = sent[i][0]
    postag = sent[i][1]
    
    features = {
        'bias': 1.0,
        'word.lower()': word.lower(),
        'word[-3:]': word[-3:],
        'word[-2:]': word[-2:],
        'word.isupper()': word.isupper(),
        'word.istitle()': word.istitle(),
        'word.isdigit()': word.isdigit(),
        'postag': postag,
        'postag[:2]': postag[:2],        
    }
    if i > 0:
        word1 = sent[i-1][0]
        postag1 = sent[i-1][1]
        features.update({
            '-1:word.lower()': word1.lower(),
            '-1:word.istitle()': word1.istitle(),
            '-1:word.isupper()': word1.isupper(),
            '-1:postag': postag1,
            '-1:postag[:2]': postag1[:2],
        })
    else:
        features['BOS'] = True
        
    if i < len(sent)-1:
        word1 = sent[i+1][0]
        postag1 = sent[i+1][1]
        features.update({
            '+1:word.lower()': word1.lower(),
            '+1:word.istitle()': word1.istitle(),
            '+1:word.isupper()': word1.isupper(),
            '+1:postag': postag1,
            '+1:postag[:2]': postag1[:2],
        })
    else:
        features['EOS'] = True
                
    return features

def sent2features(sent):
    return [word2features(sent, i) for i in range(len(sent))]

def sent2labels(sent):
    return [label for token, postag, label in sent]

def sent2tokens(sent):
    return [token for token, postag, label in sent]


if __name__ == "__main__":

    # 1.Prepare dataset
    # nltk.download('conll2002')
    # print(nltk.corpus.conll2002.fileids())
    # print(nltk.corpus.conll2002.iob_sents('esp.testa'))
    train_sents = list(nltk.corpus.conll2002.iob_sents('esp.train'))
    test_sents = list(nltk.corpus.conll2002.iob_sents('esp.testa'))

    X_train = [sent2features(s) for s in train_sents]
    y_train = [sent2labels(s) for s in train_sents]
    X_test = [sent2features(s) for s in test_sents]
    y_test = [sent2labels(s) for s in test_sents]

    #word2vec = KeyedVectors.load_word2vec_format("model/ch.300.bin", binary=True)

    # 2.Train
    crf = sklearn_crfsuite.CRF(
        algorithm='lbfgs', 
        c1=0.1, 
        c2=0.1, 
        max_iterations=100, 
        all_possible_transitions=True,
    )
    crf.fit(X_train, y_train)

    labels = list(crf.classes_)
    labels.remove('O')
    print(labels)
    
    y_pred = crf.predict(X_test)
    metrics.flat_f1_score(y_test, y_pred, 
                        average='weighted', labels=labels)

    # group B and I results
    sorted_labels = sorted(
        labels, 
        key=lambda name: (name[1:], name[0])
    )
    print(metrics.flat_classification_report(
        y_test, y_pred, labels=sorted_labels, digits=3
    ))


    # 3. Hyper parameters: c1, c2 optimization
    crf = sklearn_crfsuite.CRF(
        algorithm='lbfgs', 
        max_iterations=100, 
        all_possible_transitions=True
    )
    params_space = {
        'c1': scipy.stats.expon(scale=0.5),
        'c2': scipy.stats.expon(scale=0.05),
    }

    f1_scorer = make_scorer(metrics.flat_f1_score, 
                            average='weighted', labels=labels)

    rs = RandomizedSearchCV(crf, params_space, 
                            cv=3, 
                            verbose=2, 
                            n_jobs=-1, 
                            n_iter=10, 
                            scoring=f1_scorer)
    rs.fit(X_train, y_train)

    # result: best params
    print('best params:', rs.best_params_)
    print('best CV score:', rs.best_score_)
    print('model size: {:0.2f}M'.format(
        rs.best_estimator_.size_ / 1000000))

    # result: visualized
    _x = [s.parameters['c1'] for s in rs.grid_scores_]
    _y = [s.parameters['c2'] for s in rs.grid_scores_]
    _c = [s.mean_validation_score for s in rs.grid_scores_]

    fig = plt.figure()
    fig.set_size_inches(12, 12)
    ax = plt.gca()
    ax.set_yscale('log')
    ax.set_xscale('log')
    ax.set_xlabel('C1')
    ax.set_ylabel('C2')
    ax.set_title("Randomized Hyperparameter Search CV Results (min={:0.3}, max={:0.3})".format(
        min(_c), max(_c)
    ))

    ax.scatter(_x, _y, c=_c, s=60, alpha=0.9, edgecolors=[0,0,0])

    print("Dark blue => {:0.4}, dark red => {:0.4}".format(min(_c), max(_c)))

    # predict again with optimized params
    crf = rs.best_estimator_
    y_pred = crf.predict(X_test)
    print(metrics.flat_classification_report(
        y_test, y_pred, labels=sorted_labels, digits=3
    ))
