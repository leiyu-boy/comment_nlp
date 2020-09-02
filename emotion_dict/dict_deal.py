# -*- coding: utf-8 -*-
# @Time     : 2020/8/30 10:02 下午
# @Author   : yu.lei
import re

import pandas as pd

# 整合程度副词
from emotion_dict import CURRENT_PATH


def degree_deal():
    df1 = pd.read_csv('{}/data/degree_word_0.5.txt'.format(CURRENT_PATH), header=None, names=['word'])
    df1['weight'] = 0.5
    df2 = pd.read_csv('{}/data/degree_word_0.8.txt'.format(CURRENT_PATH), header=None, names=['word'])
    df2['weight'] = 0.8
    df3 = pd.read_csv('{}/data/degree_word_1.2.txt'.format(CURRENT_PATH), header=None, names=['word'])
    df3['weight'] = 1.2
    df4 = pd.read_csv('{}/data/degree_word_1.5.txt'.format(CURRENT_PATH), header=None, names=['word'])
    df4['weight'] = 1.5
    df5 = pd.read_csv('{}/data/degree_word_1.25.txt'.format(CURRENT_PATH), header=None, names=['word'])
    df5['weight'] = 1.25
    df6 = pd.read_csv('{}/data/degree_word_2.txt'.format(CURRENT_PATH), header=None, names=['word'])
    df6['weight'] = 2
    df = pd.concat([df1, df2, df3, df4, df5, df6])
    df.to_csv('{}/data/degree_words.csv'.format(CURRENT_PATH), index=False, encoding='utf-8')


# 由于某些程度副词、否定词、情感词会包含在停用词中，所以得过滤掉
def create_new_stopwords():
    stopwords = set()
    with open('{}/data/否定词.txt'.format(CURRENT_PATH)) as fp:
        not_word_list = fp.readlines()
    not_word_list = [i.strip() for i in not_word_list]
    degree_word = pd.read_csv('{}/data/degree_words.csv'.format(CURRENT_PATH)).word.values
    with open('{}/data/stopwords.txt'.format(CURRENT_PATH)) as fp:
        for line in fp.readlines():
            word = line.strip()
            if word not in not_word_list and word not in degree_word:
                stopwords.add(word)
    stopwords = list(stopwords)
    stopwords.append(' ')
    df = pd.DataFrame(
        {'word': stopwords}
    )
    df.to_csv('{}/data/stopwords.csv'.format(CURRENT_PATH), index=False, encoding='utf-8')


# 整合处理情感词性，来源知网
def merge_emotion():
    pos_df1 = pd.read_csv('{}/zhiwang/正面情感词语.txt'.format(CURRENT_PATH), encoding='gbk', header=None,
                          names=['word'])
    pos_df1['weight'] = 1
    pos_df2 = pd.read_csv('{}/zhiwang/正面评价词语.txt'.format(CURRENT_PATH), encoding='gbk', header=None,
                          names=['word'])
    pos_df2['weight'] = 1
    neg_df1 = pd.read_csv('{}/zhiwang/负面情感词语.txt'.format(CURRENT_PATH), encoding='gbk', header=None,
                          names=['word'])
    neg_df1['weight'] = -1
    neg_df2 = pd.read_csv('{}/zhiwang/负面评价词语.txt'.format(CURRENT_PATH), encoding='gbk', header=None,
                          names=['word'])
    neg_df2['weight'] = -1
    emotion_df = pd.concat([pos_df1, pos_df2, neg_df1, neg_df2])
    emotion_df['word'] = emotion_df.word.apply(lambda x: x.strip())
    with open('{}/data/否定词.txt'.format(CURRENT_PATH)) as fp:
        not_word_list = fp.readlines()
    not_word_list = [i.strip() for i in not_word_list]
    degree_word = list(pd.read_csv('{}/data/degree_words.csv'.format(CURRENT_PATH)).word.values)
    stop_word = list(pd.read_csv('{}/data/stopwords.csv'.format(CURRENT_PATH)).word.values)
    filter_word = not_word_list + degree_word + stop_word
    emotion_df = emotion_df[~emotion_df.word.isin(filter_word)]
    boson_df = pd.read_csv('{}/data/BosonNLP_sentiment_score.txt'.format(CURRENT_PATH), header=None, sep=' ',
                           names=['word', 'weight'])
    boson_df.dropna(subset=['word'], inplace=True)
    add_word = boson_df.loc[boson_df.word.str.contains('好|坏|差|烂|值')]
    add_word['weight'] = add_word.weight.apply(lambda x: 1 if x > 0 else -1)
    df = pd.concat([emotion_df, add_word])
    df = df.append(pd.DataFrame({'word': ['坑爹', '不值'], 'weight': [-1, -1]}), ignore_index=True)
    df.drop_duplicates(subset=['word'], inplace=True)
    df.to_csv('{}/data/emotion_word.csv'.format(CURRENT_PATH), index=False, encoding='utf-8')


def create_user_dict():
    jieba_weight = 100000
    emotion_word = list(pd.read_csv('{}/data/emotion_word.csv'.format(CURRENT_PATH)).word.values)
    stop_word = list(pd.read_csv('{}/data/stopwords.csv'.format(CURRENT_PATH)).word.values)
    with open('{}/data/否定词.txt'.format(CURRENT_PATH)) as fp:
        not_word_list = fp.readlines()
    not_word_list = [i.strip() for i in not_word_list]
    degree_word = list(pd.read_csv('{}/data/degree_words.csv'.format(CURRENT_PATH)).word.values)
    user_word = emotion_word + stop_word + not_word_list + degree_word
    user_dict = ['{} {} n'.format(i, jieba_weight) for i in user_word if len(i) > 1]
    with open('{}/data/userdict.txt'.format(CURRENT_PATH), 'w', encoding='utf-8') as fp:
        for i in user_dict:
            fp.write(i + '\n')


def emotion_dict_deal():
    degree_deal()
    create_new_stopwords()
    merge_emotion()
    create_user_dict()
