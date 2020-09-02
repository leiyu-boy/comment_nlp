# -*- coding: utf-8 -*-
# @Time     : 2020/8/21 10:46 上午
# @Author   : yu.lei
import pandas as pd

# 去除重复的评论
from reptiles_jd import CURRENT_PATH


def remove_duplicate(file):
    path = '{}/../Data/{}_comment.csv'.format(CURRENT_PATH, file)
    df = pd.read_csv(path, header=None)
    df.drop_duplicates(inplace=True)
    df.to_csv(path, index=False, header=None, encoding='utf-8')

