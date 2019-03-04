# -*- coding: utf-8 -*-
"""
@brief : 将原始数据数字化为tf特征，并将结果保存至本地
@author: Jian
"""
import time
import pickle
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer

t_start = time.time()

"""=====================================================================================================================
1 数据预处理
"""
df_train = pd.read_csv('../data/all_data.csv')
df_train.drop(columns='article', inplace=True)
df_test = pd.read_csv('../data/test_set.csv')
df_test.drop(columns='article', inplace=True)
df_all = pd.concat(objs=[df_train, df_test], axis=0, sort=True)
y_train = (df_train['class'] - 1).values

"""=====================================================================================================================
2 特征工程
"""
vectorizer = CountVectorizer(ngram_range=(1, 2), min_df=100, max_df=0.8)
x_train = vectorizer.fit_transform(df_train['word_seg'])
x_test = vectorizer.transform(df_test['word_seg'])

"""=====================================================================================================================
3 保存至本地
"""
with open('./data_tf.pkl', 'wb') as f:
    data = (x_train, y_train, x_test)
    pickle.dump(data, f)

t_end = time.time()
print("已将原始数据数字化为tf特征，共耗时：{}min".format((t_end - t_start) / 60))
