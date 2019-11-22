from itertools import chain
import sys, json, os, functools

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

def word2features(sent, i, w2v):
    word = sent[i][0]
    postag = sent[i][1]
    
    features = {
        #'bias': 1.0,
        'word': word,
        #'word.isdigit()': word.isdigit(),
        'postag': postag,
        #'postag[:1]': postag[:1],     
        #'postag[:2]': postag[:2],  
        #'len(postag)': len(postag)
    }

    if i == 0:
        features['BOS'] = True
        #features['MOS'] = False
    if i == (len(sent) - 1):
        features['EOS'] = True
        #features['MOS'] = False
    if i > 0 and (i < len(sent)-1):
        features['BOS'] = False
        features['EOS'] = False
        #features['MOS'] = True

    if len(sent) >= 2:
        if i > 0:
            word_prev = sent[i-1][0]
            postag_prev = sent[i-1][1]
            features.update({
                '-1:postag': postag_prev,
                #'-1:postag[:1]': postag_prev[:1],    
                #'-1:postag[:2]': postag_prev[:2],
                #'-1:word.isdigit()': word_prev.isdigit(),
                '-1:word': word_prev,
                #'-1:len(postag_prev)': len(postag_prev)
            })

        if i < len(sent)-1:
            word_next = sent[i+1][0]
            postag_next = sent[i+1][1]
            features.update({
                '+1:postag': postag_next,
                #'+1:postag[:1]': postag_next[:1],    
                #'+1:postag[:2]': postag_next[:2],
                #'+1:word.isdigit()': word_next.isdigit(),
                '+1:word': word_next,
                #'+1:len(postag_next)': len(postag_next)
            })
        
    if (w2v is not None) and (word in w2v.wv):
        vec_list = w2v.wv[word].tolist()
        features['word2vec_dic'] = { str(i) : val for i, val in enumerate(vec_list) }
            
    return features

def sent2features(sent, w2v):
    return [word2features(sent, i, w2v) for i in range(len(sent))]

def sent2labels(sent):
    return [label for token, postag, label in sent]

def sent2tokens(sent):
    return [token for token, postag, label in sent]


if __name__ == "__main__":

    # 1.Prepare dataset
    train_sents = None
    test_sents = None
    with open('train_char.json', encoding="utf-8") as json_file:
        train_sents = json.load(json_file)

    with open('validation_char.json', encoding="utf-8") as json_file:
        test_sents = json.load(json_file)

    # 2.Turn to features
    word2vec = KeyedVectors.load_word2vec_format("../model/ch.300.bin", binary=True)
    X_train_map = map(functools.partial(sent2features, w2v=word2vec), train_sents)
    y_train_map = map(functools.partial(sent2labels), train_sents)
    # X_test_map = map(functools.partial(sent2features, w2v=word2vec), test_sents)
    # y_test_map = map(functools.partial(sent2labels), test_sents)

    #X_train = [sent2features(s, word2vec) for s in train_sents]
    #y_train = [sent2labels(s) for s in train_sents]
    X_test = [sent2features(s, word2vec) for s in test_sents]
    y_test = [sent2labels(s) for s in test_sents]


    # 3.Train
    crf = sklearn_crfsuite.CRF(
        algorithm='lbfgs', 
        c1=0.1, 
        c2=0.1, 
        max_iterations=100, 
        all_possible_transitions=True,
    )
    crf.fit(list(X_train_map), list(y_train_map))

    labels = list(crf.classes_)
    labels.remove('O')
    print(labels)
    
    # 4.Validate
    y_pred = crf.predict(X_test)
    metrics.flat_f1_score(y_test, y_pred, 
                        average='weighted', labels=labels)

    sorted_labels = sorted(
        labels, 
        key=lambda name: (name[1:], name[0])
    )
    print(metrics.flat_classification_report(
        y_test, y_pred, labels=sorted_labels, digits=3
    ))


    # 5. Hyper parameters: c1, c2 optimization
    # crf = sklearn_crfsuite.CRF(
    #     algorithm='lbfgs', 
    #     max_iterations=100, 
    #     all_possible_transitions=True
    # )
    # params_space = {
    #     'c1': scipy.stats.expon(scale=0.5),
    #     'c2': scipy.stats.expon(scale=0.05),
    # }

    # f1_scorer = make_scorer(metrics.flat_f1_score, 
    #                         average='weighted', labels=labels)

    # rs = RandomizedSearchCV(crf, params_space, 
    #                         cv=3, 
    #                         verbose=2, 
    #                         n_jobs=-1, 
    #                         n_iter=10, 
    #                         scoring=f1_scorer)
    # rs.fit(X_train, y_train)

    # # result: best params
    # print('best params:', rs.best_params_)
    # print('best CV score:', rs.best_score_)
    # print('model size: {:0.2f}M'.format(
    #     rs.best_estimator_.size_ / 1000000))

    # # result: visualized
    # _x = [s['c1'] for s in rs.cv_results_['params']]
    # _y = [s['c2'] for s in rs.cv_results_['params']]
    # _c = [s for s in rs.cv_results_['mean_test_score']]

    # fig = plt.figure()
    # fig.set_size_inches(12, 12)
    # ax = plt.gca()
    # ax.set_yscale('log')
    # ax.set_xscale('log')
    # ax.set_xlabel('C1')
    # ax.set_ylabel('C2')
    # ax.set_title("Randomized Hyperparameter Search CV Results (min={:0.3}, max={:0.3})".format(
    #     min(_c), max(_c)
    # ))

    # ax.scatter(_x, _y, c=_c, s=60, alpha=0.9, edgecolors=[0,0,0])

    # print("Dark blue => {:0.4}, dark red => {:0.4}".format(min(_c), max(_c)))

    # # predict again with optimized params
    # crf = rs.best_estimator_
    # y_pred = crf.predict(X_test)
    # print(metrics.flat_classification_report(
    #     y_test, y_pred, labels=sorted_labels, digits=3
    # ))
