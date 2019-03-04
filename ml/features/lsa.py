# -*- coding: utf-8 -*-
"""
@brief : 将tfidf特征降维为lsa特征，并将结果保存至本地
"""
from sklearn.decomposition import TruncatedSVD
import pickle
import time


def feature_lsa():
    t1 = time.time()

    print('加载feature_tfidf_select。。。')
    tfidf_path = '../data/tmp/feature_tfidf_select_LSVC_l2_675311.pkl'
    with open(tfidf_path, 'rb') as f:
        X_train, y_train, X_test = pickle.load(f)
    t2 = time.time()
    print('加载feature_tfidf_select用时：{}s'.format(t2 - t1))

    print("抽取lsa特征。。。")
    lsa = TruncatedSVD(n_components=200)
    X_train = lsa.fit_transform(X_train)
    X_test = lsa.transform(X_test)
    t3 = time.time()
    print('抽取lsa特征用时：{}s'.format(t3 - t2))

    print('持久化数据。。。')
    with open('../data/tmp/feature_tfidf_select_lsa.pkl', 'wb') as f:
        data = (X_train, y_train, X_test)
        pickle.dump(data, f)
    t4 = time.time()
    print('持久化数据用时：{}s'.format(t4 - t3))

    print("共用时：{}s".format(t4 - t1))


if __name__ == '__main__':
    feature_lsa()
