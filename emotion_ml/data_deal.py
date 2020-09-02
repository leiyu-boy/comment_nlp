# -*- coding: utf-8 -*-
# @Time     : 2020/9/1 6:02 下午
# @Author   : yu.lei
import pickle

import pandas as pd
from gensim.models import Word2Vec
from scipy.sparse import coo_matrix
from sklearn.feature_extraction.text import TfidfTransformer, CountVectorizer
from sklearn.linear_model import LogisticRegression

from emotion_dict.dict_analy import filter_stop_words
from emotion_ml import CURRENT_PATH
import numpy as np


# 构建词向量，得到训练集
def get_word_vec_model():
    df1 = pd.read_csv('{}/../Data/deal_bad_comment.csv'.format(CURRENT_PATH), header=None, names=['word'])
    df2 = pd.read_csv('{}/../Data/deal_great_comment.csv'.format(CURRENT_PATH), header=None, names=['word'])
    df1['y'] = 1
    df2['y'] = 0
    df = pd.concat([df1, df2])
    # 删除字母
    df['word'] = df.word.str.replace('\n| |[a-zA-Z0-9]|&', ',')
    df['word'] = df['word'].apply(lambda x: filter_stop_words(x))
    vec_model = Word2Vec(size=200, min_count=5)
    x = df['word']
    vec_model.build_vocab(x)
    vec_model.train(x, total_examples=vec_model.corpus_count, epochs=vec_model.iter)
    vec_model.save('{}/models/word_vec'.format(CURRENT_PATH))
    train_x = np.concatenate([seg_vec(vec_model, word) for word in x])
    with open('{}/data/train_x.pkl'.format(CURRENT_PATH), 'wb') as fp:
        pickle.dump(train_x, fp)
    with open('{}/data/train_y.pkl'.format(CURRENT_PATH), 'wb') as fp:
        pickle.dump(df['y'], fp)


# 计算句子的向量
def seg_vec(vec_model, word):
    vec = np.zeros(200).reshape((1, 200))
    for i in word:
        try:
            vec += vec_model.wv[i].reshape((1, 200))
        except:
            continue
    return vec

