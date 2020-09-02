# -*- coding: utf-8 -*-
# @Time     : 2020/9/2 12:08 上午
# @Author   : yu.lei
import pickle

import jieba
import joblib
import numpy as np
import pandas as pd
# 逻辑回归
from gensim.models import Word2Vec
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

from emotion_ml import CURRENT_PATH
from emotion_ml.data_deal import seg_vec


def logistic():
    X = np.loadtxt('{}/data/word_fre.txt'.format(CURRENT_PATH))
    y = pd.read_csv('{}/data/split_word.csv'.format(CURRENT_PATH)).word
    print(X)
    LR = LogisticRegression(solver='liblinear')
    LR.fit(X, y)


def svm():
    with open('{}/data/train_x.pkl'.format(CURRENT_PATH), 'rb') as fp:
        train_X = pickle.load(fp)
    with open('{}/data/train_y.pkl'.format(CURRENT_PATH), 'rb') as fp:
        train_y = pickle.load(fp)
    model = SVC(kernel='rbf')
    model.fit(train_X, train_y)
    with open('{}/models/svm_model.pkl'.format(CURRENT_PATH), 'wb') as fp:
        pickle.dump(model, fp)


def result_predict(ml_type):
    vec_model = Word2Vec.load('{}/models/word_vec'.format(CURRENT_PATH))
    model = joblib.load('{}/models/{}_model.pkl'.format(CURRENT_PATH, ml_type))
    great_df = pd.read_csv('{}/../Data/test_great_comment.csv'.format(CURRENT_PATH), header=None, names=['word']).head(
        10)
    bad_df = pd.read_csv('{}/../Data/test_bad_comment.csv'.format(CURRENT_PATH), header=None, names=['word']).head(10)
    great_df['y'] = 0
    bad_df['y'] = 1
    df = pd.concat([great_df, bad_df])
    df['pre_y'] = df.word.apply(lambda x: int(model.predict(seg_vec(vec_model, jieba.lcut(x)))))
    df.to_csv('{}/../Data/svm_comment.csv'.format(CURRENT_PATH), index=False, encoding='utf-8')


if __name__ == '__main__':
    # logistic()
    svm()
    result_predict('svm')
