# -*- coding: utf-8 -*-
# @Time     : 2020/8/21 10:49 上午
# @Author   : yu.lei

# 准备数据
import re

import jieba
import joblib
from gensim.models import Word2Vec
from sklearn.model_selection import train_test_split

from emotion_dict.dict_analy import filter_stop_words, match_word, cal_score
from emotion_dict.dict_deal import emotion_dict_deal
from emotion_ml.data_deal import get_word_vec_model, seg_vec
from emotion_ml.train import model_train
from reptiles_jd.deal import remove_duplicate
from reptiles_jd.reptile import wares_search
import pandas as pd


# 准备数据
def prepare_data():
    # 爬取数据
    keyword_list = ['鞋子', '衬衣', '船袜', '相机', '书', '裤子', '数据线', '洗发水', '微波炉', '水杯', '置物架',
                    '口红', '皮带', '背包', '糕点模具', '山地车', '方便面']
    for i in keyword_list:
        wares_search(i)
    # 去重
    for i in ['great', 'bad']:
        remove_duplicate(i)
        remove_duplicate(i)


# 各取25%进行测试,75%进行分析处理
def split_data():
    df1 = pd.read_csv('Data/bad_comment.csv')
    df2 = pd.read_csv('Data/great_comment.csv')
    # 数据爬的有点多，为节省时间，只取80%的数据
    df1, _ = train_test_split(df1, test_size=0.2)
    df2, _ = train_test_split(df2, test_size=0.2)
    deal_df1, test_df1 = train_test_split(df1, test_size=0.25)
    deal_df2, test_df2 = train_test_split(df2, test_size=0.25)
    test_df1.to_csv('Data/test_bad_comment.csv', index=False, encoding='utf-8')
    deal_df1.to_csv('Data/deal_bad_comment.csv', index=False, encoding='utf-8')
    test_df2.to_csv('Data/test_great_comment.csv', index=False, encoding='utf-8')
    deal_df2.to_csv('Data/deal_great_comment.csv', index=False, encoding='utf-8')


'''
如果需要重新整理词典，则boolean_dict_deal=True
border为分类阈值
seg为单个句子
df为待分析的文件
all_score为情感分
pre_y为分类结果，1为好评，0为正差评
'''


def emotion_dict_analy(seg, boolean_dict_deal=False, border=0):
    print(seg)
    if boolean_dict_deal:
        emotion_dict_deal()
    seg_list = re.split('，|；|。', seg)
    all_score = 0
    for seg in seg_list:
        word_list = filter_stop_words(seg)
        word, index, weight = match_word(word_list)
        score = cal_score(word_list, word, index, weight)
        all_score += score
    if score < border:
        pre_y = 1
    else:
        pre_y = 0
    print(all_score)
    return all_score, pre_y


'''
ml_type为模型训练采用的算法
是否需要重新训练词向量，如果需要，则boolean_train_vec_model=True
是否需要重新训练模型，如果需要，则boolean_train=True
seg为待分析的句子
return:分类值，1为差评，0为好评
'''


def emotion_ml_analy(seg, vec_model, model):
    word_vec = seg_vec(vec_model, jieba.lcut(seg))
    if model in ['knn', 'NB']:
        standard = joblib.load('emotion_ml/models/standard_model.pkl')
        word_vec = standard.transform(word_vec)
    return int(model.predict(word_vec))


class ResultAnaly(object):
    def __init__(self, analy_method):
        df = pd.read_csv('Data/{}_comment.csv'.format(analy_method))
        self.data = df

    # 准确率
    def precision(self):
        data = self.data
        return data[data.y == data.pre_y].shape[0] / data.shape[0]

    # 坏评召回率
    def recall(self):
        data = self.data
        return data[(data.pre_y == 0) & (data.y == 0)].shape[0] / data[data.y == 0].shape[0]


if __name__ == '__main__':
    # 若已有数据，注释掉下面的一行即可
    # prepare_data()
    # split_data()
    # 单个句子
    seg = '非常不好，完全不值这个价，太坑爹了'
    # 文件
    great_df = pd.read_csv('Data/test_great_comment.csv', header=None, names=['word'])
    bad_df = pd.read_csv('Data/test_bad_comment.csv', header=None, names=['word'])
    great_df['y'] = 0
    bad_df['y'] = 1
    df = pd.concat([great_df, bad_df])
    # 基于情感词典的分析
    # df[['score', 'pre_y']] = df.apply(lambda x: emotion_dict_analy(x.word), axis=1, result_type='expand')
    # df.to_csv('Data/dict_comment.csv', index=False, encoding='utf-8')
    # exit()
    # 基于机器学习
    ml_types = ['NB', 'knn', 'xgb']
    for ml_type in ml_types:
        print(ml_type)
        model_train(ml_type)
        vec_model = Word2Vec.load('emotion_ml/models/word_vec')
        model = joblib.load('emotion_ml/models/{}_model.pkl'.format(ml_type))
        df['pre_y'] = df.word.apply(lambda x: emotion_ml_analy(x, vec_model, model))
        df.to_csv('Data/{}_comment.csv'.format(ml_type), index=False, encoding='utf-8')
        result_analy = ResultAnaly(ml_type)
        print(result_analy.precision(), result_analy.recall())
    # pass
