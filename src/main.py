from itertools import chain
import sys, json, os, functools, itertools

# model
import gensim
from gensim.models import Word2Vec
from gensim.models import KeyedVectors

# machine learning
import sklearn
import scipy.stats
from sklearn.externals import joblib
from sklearn.metrics import make_scorer
from sklearn.model_selection import train_test_split
from sklearn.model_selection import RandomizedSearchCV
import sklearn_crfsuite
from sklearn_crfsuite import scorers
from sklearn_crfsuite import metrics


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

def getMapChunk(_map, start, length):
    return itertools.islice(_map, start, start + length)

def trainAndValidate(X_train_map, y_train_map, X_test, y_test, existed_model=None, index=0, should_dump=True):
    # train
    if existed_model is None:
        crf = sklearn_crfsuite.CRF(
            algorithm='lbfgs', 
            c1=0.1, 
            c2=0.1, 
            max_iterations=100, 
            all_possible_transitions=True,
        )
    else:
        crf = joblib.load(existed_model)

    crf.fit(list(X_train_map), list(y_train_map))
    if should_dump:
        joblib.dump(crf, 'crf_model_' + str(index) + '.pkl')

    # validate
    y_pred = crf.predict(X_test)
    labels = list(crf.classes_)
    labels.remove('O')
    sorted_labels = sorted(
        labels, 
        key=lambda name: (name[1:], name[0])
    )
    print(metrics.flat_classification_report(
        y_test, y_pred, labels=sorted_labels, digits=3
    ))

    
if __name__ == "__main__":

    # 1.Prepare dataset
    with open('train_char.json', encoding="utf-8") as json_file:
        train_sents = json.load(json_file)

    with open('validation_char.json', encoding="utf-8") as json_file:
        test_sents = json.load(json_file)
    len_train = len(train_sents)

    # 2.Turn to features
    word2vec = KeyedVectors.load_word2vec_format("../model/ch.300.bin", binary=True)
    X_train_map = map(functools.partial(sent2features, w2v=word2vec), train_sents)
    y_train_map = map(functools.partial(sent2labels), train_sents)
    X_test = [sent2features(s, word2vec) for s in test_sents]
    y_test = [sent2labels(s) for s in test_sents]

    # 3.Split x, y into chunks to alliviate memory use
    n = 10
    length = int( (len_train - len_train % n) / n )

    # 4.Repeat train & validate for each chunk
    print('='*20 + 'chunk 0' + '='*20)
    trainAndValidate(
        getMapChunk(X_train_map, 0, length),
        getMapChunk(y_train_map, 0, length),
        X_test,
        y_test,
    )

    for i in range(1, n):
        print('='*20 + 'chunk ' + str(i) + '='*20)
        trainAndValidate(
            getMapChunk(X_train_map, i*length, length),
            getMapChunk(y_train_map, i*length, length),
            X_test,
            y_test,
            'crf_model_' + str(i-1) + '.pkl',
            i,
        )

   
