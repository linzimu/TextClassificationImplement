# -*- coding: utf-8 -*-
"""
@brief : lgb算法
"""
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
import time
import pickle
import lightgbm as lgb


def f1_score_vali(preds, data_vali):
    labels = data_vali.get_label()
    preds = np.argmax(preds.reshape(20, -1), axis=0)
    score_vali = f1_score(y_true=labels, y_pred=preds, average='macro')
    return 'f1_score', score_vali, True


def train_model():
    t1 = time.time()
    print("读取数据。。。")
    features_path = '../data/tmp/feature_ensemble.pkl'
    with open(features_path, 'rb') as f:
        X_train, y_train, X_test = pickle.load(f)
    X_train, X_vali, y_train, y_vali = train_test_split(X_train, y_train, test_size=0.1, random_state=0)
    d_train = lgb.Dataset(data=X_train, label=y_train)
    d_vali = lgb.Dataset(data=X_vali, label=y_vali)
    t2 = time.time()
    print('读取数据用时：{}s'.format(t2 - t1))

    print('训练模型。。。')
    params = {
        'boosting': 'gbdt',  # 该参数为默认参数
        'application': 'multiclass',  # 分类任务
        'num_class': 20,
        'learning_rate': 0.1,  # 学习率；默认值为0.1
        'num_leaves': 31,  # 一棵树中的最大叶子数；默认值为31
        'max_depth': -1,  # 限制树的最大深度；默认值-1表示没有限制
        'lambda_l1': 0,  # L1正则化；默认0.0表示没有约束
        'lambda_l2': 0.5,  # L2正则化；默认0.0表示没有约束
        # 每次迭代中用的数据比列；默认值1.0
        # 用来加速训练以及防止过拟合；
        'bagging_fraction': 0.8,  #
        # 每次迭代中随机选择部分特征；默认值1.0
        # 用来加速训练过程以及过拟合；
        'feature_fraction': 0.8
    }
    bst = lgb.train(params, d_train, num_boost_round=800, valid_sets=d_vali, feval=f1_score_vali,
                    early_stopping_rounds=None,
                    verbose_eval=True)
    t3 = time.time()
    # 保存模型
    bst.save_model('./model.txt')
    print('训练模型并保存模型用时：{}s'.format(t3 - t2))

    print("共用时：{}s".format(t3 - t1))


def predict():
    t1 = time.time()

    print('加载数据。。。')
    features_path = '../data/tmp/feature_ensemble.pkl'
    with open(features_path, 'rb') as f:
        _, _, X_test = pickle.load(f)
    with open('../data/test_data.pkl', 'rb') as f:
        _, y_test = pickle.load(f)
    bst = lgb.Booster(model_file='./model.txt')
    t2 = time.time()
    print('加载数据用时：{}s'.format(t2 - t1))

    print('预测。。。')
    y_pre = np.argmax(bst.predict(X_test), axis=1)
    print('模型预测正确率：', sum(y_test == y_pre) / len(y_test))
    t3 = time.time()
    print('预测用时：{}'.format(t3 - t2))

    print("共用时：{}s".format(t3 - t1))


if __name__ == '__main__':
    # train_model()
    predict()
    """
        加载数据。。。
        加载数据用时：5.556754112243652s
        预测。。。
        模型预测正确率： 0.7616999087472298
        预测用时：37.887290954589844
        共用时：43.444045066833496s
    """
