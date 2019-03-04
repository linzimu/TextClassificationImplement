# -*- coding: utf-8 -*-
"""
@简介：对特征进行嵌入式选择
"""
import time
import pickle
from sklearn.feature_selection import SelectFromModel
from sklearn.svm import LinearSVC


def feature_select():
    t1 = time.time()

    print('加载数据。。。')
    features_path = '../data/tmp/feature_tfidf.pkl'  # tfidf特征的路径
    with open(features_path, 'rb') as f:
        X_train, y_train, X_test = pickle.load(f)
    t2 = time.time()
    print('加载数据用时：{}s'.format(t2 - t1))

    print('特征选择。。。')
    lsvc = LinearSVC(penalty='l2', C=1.0, dual=True).fit(X_train, y_train)
    slt = SelectFromModel(lsvc, prefit=True)
    # X_train_s = slt.transform(X_train)  # 这里是不是有问题？？？
    X_train_s = slt.fit_transform(X_train)
    X_test_s = slt.transform(X_test)
    t3 = time.time()
    print('特征选择用时：{}s'.format(t3 - t2))

    print('数据持久化。。。')
    alo_name = 'LSVC_l2'
    print(X_train_s.shape)
    num_features = X_train_s.shape[1]
    data_path = features_path.rsplit('.', 1)[0] + '_select_' + alo_name + '_' + str(num_features) + '.pkl'
    with open(data_path, 'wb') as f:
        pickle.dump((X_train_s, y_train, X_test_s), f)
    t4 = time.time()
    print('数据持久化用时：{}s'.format(t4 - t3))

    print('总用时：{}s'.format(t4 - t1))


if __name__ == '__main__':
    feature_select()
