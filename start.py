# -*- coding: utf-8 -*-
# @Time     : 2020/8/21 10:49 上午
# @Author   : yu.lei

# 准备数据
import re

from sklearn.model_selection import train_test_split

from emotion_dict.dict_analy import filter_stop_words, match_word, cal_score
from emotion_dict.dict_deal import emotion_dict_deal
from emotion_ml.data_deal import get_word_vec_model
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


def dup():
    df1 = pd.read_csv('Data/bad_comment.csv', header=None, names=['word'])
    df2 = pd.read_csv('Data/great_comment.csv', header=None, names=['word'])
    print(df1.word.value_counts())
    print(df2.word.value_counts())


# 如果需要重新整理词典，则boolean_dict_deal=True
def emotion_dict_analy(boolean_dict_deal=False):
    if boolean_dict_deal:
        emotion_dict_deal()

    # 单独分析某个句子
    # sentence = '非常不好，完全不值这个价，太坑爹了'
    # seg_list = re.split('，|；|。', sentence)
    # print(seg_list)
    # all_score = 0
    # for seg in seg_list:
    #     word_list = filter_stop_words(seg)
    #     print(word_list)
    #     word, index, weight = match_word(word_list)
    #     print(word, index, weight)
    #     score = cal_score(word_list, word, index, weight)
    #     print(score)
    #     all_score += score
    # print(all_score)

    # 对测试文件批量分析
    def get_score(x):
        seg_list = re.split('，|；|。', x)
        all_score = 0
        for seg in seg_list:
            word_list = filter_stop_words(seg)
            word, index, weight = match_word(word_list)
            score = cal_score(word_list, word, index, weight)
            all_score += score
        print(x, all_score)
        return all_score

    great_df = pd.read_csv('Data/test_great_comment.csv', header=None, names=['word'])
    bad_df = pd.read_csv('Data/test_bad_comment.csv', header=None, names=['word'])
    great_df['y'] = 0
    bad_df['y'] = 1
    df = pd.concat([great_df, bad_df])
    df['pre_y'] = df.word.apply(lambda x: 1 if get_score(x) < 0 else 0)
    df.to_csv('Data/dict_comment.csv', index=False, encoding='utf-8')


# 是否需要重新训练词向量，如果需要，则boolean_train_vec_model=True
def emotion_ml_analy(ml_type, boolean_train_vec_model=False):
    # 得到词向量模型、训练集，如果已存在，注释下面一行
    get_word_vec_model()


class ResultAnaly(object):
    def __init__(self, analy_method):
        df = pd.read_csv('Data/{}_comment.csv'.format(analy_method))
        self.data = df

    # 准确率
    def precision(self):
        data = self.data
        return data[data.y == data.pre_y].shape[0] / data.shape[0]

    # 好评召回率
    def recall(self):
        data = self.data
        return data[(data.pre_y == 0) & (data.y == 0)].shape[0] / data[data.y == 0].shape[0]


def look_data():
    comment = ['bad', 'great']
    for i in comment:
        df1 = pd.read_csv('Data/{}_comment.csv'.format(i), header=None)
        df2 = pd.read_csv('Data/deal_{}_comment.csv'.format(i), header=None)
        print(df1.shape[0], df2.shape[0])


if __name__ == '__main__':
    # 若已有数据，注释掉下面的一行即可
    # prepare_data()
    # split_data()
    # result_analy('dict')
    # look_data()
    # emotion_ml_analy('svm')
    dict_analy = ResultAnaly('dict')
    print(dict_analy.precision(), dict_analy.recall())
    # pass
