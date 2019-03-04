# -*- coding: utf-8 -*-
"""
@brief : lda/lsa/doc2vec三种特征进行特征融合，并将结果保存至本地
"""
import numpy as np
import pickle
import time


def feature_fusion():
    t1 = time.time()

    print('加载lda、lsa、doc2vec特征，同时进行特征融合。。。')
    with open('../data/tmp/feature_tfidf_select_lda.pkl', 'rb') as f:
        X_train_1, y_train, X_test_1 = pickle.load(f)
    with open('../data/tmp/feature_tfidf_select_lsa.pkl', 'rb') as f:
        X_train_2, _, X_test_2 = pickle.load(f)
    with open('../data/tmp/feature_doc2vec_25.pkl', 'rb') as f:
        X_train_3, _, X_test_3 = pickle.load(f)
    # 横向合并特征构成新的特征
    X_train = np.concatenate((X_train_1, X_train_2, X_train_3), axis=1)
    X_test = np.concatenate((X_test_1, X_test_2, X_test_3), axis=1)
    t2 = time.time()
    print('加载lda、lsa、doc2vec特征，同时进行特征融合用时：{}s'.format(t2 - t1))

    print('持久化数据。。。')
    with open('../data/tmp/feature_ensemble.pkl', 'wb') as f:
        data = (X_train, y_train, X_test)
        pickle.dump(data, f)
    t3 = time.time()
    print('持久化数据共用时：{}s'.format(t3 - t2))

    print('共用时：{}s'.format(t3 - t1))


if __name__ == '__main__':
    feature_fusion()
