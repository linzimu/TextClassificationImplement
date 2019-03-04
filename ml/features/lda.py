# -*- coding: utf-8 -*-
"""
@brief : 将tf特征降维为lda特征，并将结果保存至本地
"""
from sklearn.decomposition import LatentDirichletAllocation
import pickle
import time


def feature_lda():
    t1 = time.time()

    print('加载feature_tfidf_select。。。')
    tfidf_path = '../data/tmp/feature_tfidf_select_LSVC_l2_675311.pkl'
    with open(tfidf_path, 'rb') as f:
        X_train, y_train, X_test = pickle.load(f)
    t2 = time.time()
    print('加载feature_tfidf_select用时：{}s'.format(t2 - t1))

    print("抽取lda特征。。。")
    lda = LatentDirichletAllocation(n_components=200)
    X_train = lda.fit_transform(X_train)
    X_test = lda.transform(X_test)
    t3 = time.time()
    print('抽取lda特征用时：{}s'.format(t3 - t2))

    print('持久化数据。。。')
    with open('../data/tmp/feature_tfidf_select_lda.pkl', 'wb') as f:
        data = (X_train, y_train, X_test)
        pickle.dump(data, f)
    t4 = time.time()
    print('持久化数据用时：{}s'.format(t4 - t3))

    print("共用时：{}s".format(t4 - t1))


if __name__ == '__main__':
    feature_lda()
    # 加载feature_tfidf_select。。。
    # 加载feature_tfidf_select用时：6.423671007156372s
    # 抽取lda特征。。。
    # 抽取lda特征用时：10675.465195178986s
    # 持久化数据。。。
    # 持久化数据用时：1.3382072448730469s
    # 共用时：10683.227073431015s
