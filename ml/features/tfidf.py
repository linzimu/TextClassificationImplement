# -*- coding: utf-8 -*-
"""
@brief : 将原始数据数字化为tfidf特征，并将结果保存至本地
"""
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle
import time


def get_tfidf_feature():
    t1 = time.time()

    print('加载数据。。。')
    with open('../data/test_data.pkl', 'rb') as f:
        X_test, y_test = pickle.load(f)
    with open('../data/train_data.pkl', 'rb') as f:
        X_train, y_train = pickle.load(f)
    t2 = time.time()
    print('加载数据用时：{}s'.format(t2 - t1))

    # =====================================================================================================================
    # 2 特征工程-提取tfidf特征
    # sublinear_tf 取值True表示用1+log(tf)表示tf
    print('提取tfidf特征。。。')
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=3, max_df=0.9, sublinear_tf=True)
    X_train = vectorizer.fit_transform(X_train)
    X_test = vectorizer.transform(X_test)
    t3 = time.time()
    print('提取tfidf特征用时：{}s'.format(t3 - t2))

    # =====================================================================================================================
    # 3 保存至本地
    print('持久化数据。。。')
    with open('../data/tmp/feature_tfidf.pkl', 'wb') as f:
        pickle.dump((X_train, y_train, X_test), f)
    t4 = time.time()
    print('持久化数据用时：{}s'.format(t4 - t3))

    print('总用时：{}s'.format(t4 - t1))


if __name__ == '__main__':
    """
        加载数据。。。
        加载数据用时：5.842896223068237s
        提取tfidf特征。。。
        提取tfidf特征用时：286.7960388660431s
        持久化数据。。。
        持久化数据用时：8.780410528182983s
        总用时：301.4193456172943s
    """
    get_tfidf_feature()
