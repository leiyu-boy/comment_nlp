# -*- coding: utf-8 -*-
# @Time     : 2020/8/30 6:58 下午
# @Author   : yu.lei
import re
import jieba
import pandas as pd

from emotion_dict import CURRENT_PATH


# jieba分词，过滤掉停用词
def filter_stop_words(sentence):
    stop_words = pd.read_csv('{}/data/stopwords.csv'.format(CURRENT_PATH)).word.values
    # 引用自定义分词
    jieba.load_userdict(r'{}/data/userdict.txt'.format(CURRENT_PATH))
    seg_list = jieba.lcut(sentence, cut_all=False)
    return [i for i in seg_list if i not in stop_words]


# 标记情感词的位置
def match_word(word_list):
    emotion_df = pd.read_csv('{}/data/emotion_word.csv'.format(CURRENT_PATH))
    emotion_df.set_index('word', inplace=True)
    emotion_dict = emotion_df['weight'].to_dict()
    seg_emotion_word = []
    emotion_index = []
    emotion_weight = []
    for index, word in enumerate(word_list):
        if word in emotion_dict.keys():
            seg_emotion_word.append(word)
            emotion_index.append(index)
            emotion_weight.append(emotion_dict[word])
    return seg_emotion_word, emotion_index, emotion_weight


def cal_score(word_list, seg_emotion_word, emotion_index, emotion_weight):
    score = 0
    if len(seg_emotion_word) == 0:
        return score
    degree_df = pd.read_csv('{}/data/degree_words.csv'.format(CURRENT_PATH))
    degree_df.set_index('word', inplace=True)
    degree_dict = degree_df['weight'].to_dict()
    with open('{}/data/否定词.txt'.format(CURRENT_PATH)) as fp:
        not_word_list = fp.readlines()
    not_word_list = [i.strip() for i in not_word_list]
    # 匹配两个情感词之间的否定词、程度副词
    for i in range(0, len(emotion_index)):
        # 单独处理第一个情感词前面的否定词、程度副词
        if i == 0:
            not_w = 1
            degree_w = 1
            for j in range(0, emotion_index[i]):
                if word_list[j] in not_word_list:
                    not_w *= -1
                elif word_list[j] in degree_dict.keys():
                    degree_w *= degree_dict[word_list[j]]
            score += emotion_weight[i] * not_w * degree_w
        if i == len(emotion_index) - 1:
            return score
        not_w = 1
        degree_w = 1
        for j in range(emotion_index[i] + 1, emotion_index[i + 1]):
            if word_list[j] in not_word_list:
                not_w *= -1
            elif word_list[j] in degree_dict.keys():
                degree_w *= degree_dict[word_list[j]]
        score += emotion_weight[i + 1] * not_w * degree_w
    return score


# 批量计算分数，确定临界值
def batch_cal_score():
    comment_type = ['bad', 'great']

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

    for i in comment_type:
        path = '{}/../Data/test_{}_comment.csv'.format(CURRENT_PATH, i)
        df = pd.read_csv(path, header=None, names=['word'])
        df['score'] = df.word.apply(lambda x: get_score(x))
        df.to_csv('{}/../Data/dict_{}_comment_score.csv'.format(CURRENT_PATH, i), index=False, encoding='utf-8')

