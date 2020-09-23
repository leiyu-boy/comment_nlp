# -*- coding: utf-8 -*-
# @Time     : 2020/9/2 12:08 上午
# @Author   : yu.lei
import pickle
from sklearn import neighbors
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC
from xgboost import XGBClassifier

from emotion_ml import CURRENT_PATH
import tensorflow as tf


# 逻辑回归
def logistic(train_X, train_y):
    model = LogisticRegression(solver='liblinear')
    model.fit(train_X, train_y)
    with open('{}/models/logistic_model.pkl'.format(CURRENT_PATH), 'wb') as fp:
        pickle.dump(model, fp)


# 支持向量机
def svm(train_X, train_y):
    model = SVC(kernel='rbf')
    model.fit(train_X, train_y)
    with open('{}/models/svm_model.pkl'.format(CURRENT_PATH), 'wb') as fp:
        pickle.dump(model, fp)


# 随机森林
def random_forest(train_X, train_y):
    model = RandomForestClassifier(n_estimators=20)
    model.fit(train_X, train_y)
    with open('{}/models/random_forest_model.pkl'.format(CURRENT_PATH), 'wb') as fp:
        pickle.dump(model, fp)


# 朴素贝叶斯
def NB(train_X, train_y):
    model = MultinomialNB()
    model.fit(train_X, train_y)
    with open('{}/models/NB_model.pkl'.format(CURRENT_PATH), 'wb') as fp:
        pickle.dump(model, fp)


# 最近邻
def knn(train_X, train_y):
    model = neighbors.KNeighborsClassifier(n_neighbors=2)
    model.fit(train_X, train_y)
    with open('{}/models/knn_model.pkl'.format(CURRENT_PATH), 'wb') as fp:
        pickle.dump(model, fp)


def xgb(train_X, train_y):
    model = XGBClassifier()
    model.fit(train_X, train_y)
    with open('{}/models/xgb_model.pkl'.format(CURRENT_PATH), 'wb') as fp:
        pickle.dump(model, fp)


def mlp_model(train_X, train_y):
    model = MLPClassifier()
    model.fit(train_X, train_y)
    with open('{}/models/mlp_model.pkl'.format(CURRENT_PATH), 'wb') as fp:
        pickle.dump(model, fp)


def lstm_model(train_X, train_y):
    batchSize = 24
    lstmUnits = 64
    numClasses = 2
    iterations = 50000
    tf.reset


def model_train(ml_type):
    with open('{}/data/train_x.pkl'.format(CURRENT_PATH), 'rb') as fp:
        train_X = pickle.load(fp)
    with open('{}/data/train_y.pkl'.format(CURRENT_PATH), 'rb') as fp:
        train_y = pickle.load(fp)
    standardScaler = MinMaxScaler(feature_range=(0, 10))
    standardScaler.fit(train_X)
    x_train_standard = standardScaler.transform(train_X)
    with open('{}/models/standard_model.pkl'.format(CURRENT_PATH), 'wb') as fp:
        pickle.dump(standardScaler, fp)
    if ml_type == 'logistic':
        logistic(train_X, train_y)
    elif ml_type == 'svm':
        svm(train_X, train_y)
    elif ml_type == 'random_forest':
        random_forest(train_X, train_y)
    elif ml_type == 'NB':
        NB(x_train_standard, train_y)
    elif ml_type == 'knn':
        knn(train_X, train_y)
    elif ml_type == 'xgb':
        xgb(train_X, train_y)
    elif ml_type == 'mlp':
        mlp_model(train_X, train_y)
