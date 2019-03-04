# -*- coding: utf-8 -*-
"""
@brief : 将原始数据数字化为doc2vec特征，并将结果保存至本地
"""
import pandas as pd
import numpy as np
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
import time
import pickle


def sentence2list(sentence):
    s_list = sentence.strip().split()
    return s_list


def feature_doc2vec():
    t1 = time.time()
    print('加载数据和分词。。。')
    with open('../data/test_data.pkl', 'rb') as f:
        X_test, y_test = pickle.load(f)
    with open('../data/train_data.pkl', 'rb') as f:
        X_train, y_train = pickle.load(f)
    # axis, 取0：纵向合并；取1：横向合并
    X_all = pd.concat(objs=[X_train, X_test], axis=0, sort=True)
    X_all['word_list'] = X_all.apply(sentence2list)
    texts = X_all['word_list'].tolist()
    t2 = time.time()
    print('加载数据和分词用时：{}s'.format(t2 - t1))

    print('抽取句子特征。。。')
    # 这里将测试集和训练集一起做句子特征提取
    documents = [TaggedDocument(doc, [i]) for i, doc in enumerate(texts)]
    model = Doc2Vec(documents, vector_size=200, window=5, min_count=3, workers=4, epochs=25)
    docvecs = model.docvecs
    X_train, X_test = [], []
    for i in range(0, 102277):
        if i < 71593:
            X_train.append(docvecs[i])
        else:
            X_test.append(docvecs[i])
    X_train = np.array(X_train)
    X_test = np.array(X_test)
    t3 = time.time()
    print('抽取句子特种用时：{}s'.format(t3 - t2))

    print('持久化数据。。。')
    with open('../data/tmp/feature_doc2vec_25.pkl', 'wb') as f:
        data = (X_train, y_train, X_test)
        pickle.dump(data, f)
    t4 = time.time()
    print('持久化数据用时：{}s'.format(t4 - t3))

    print('总用时：{}s'.format(t4 - t1))


if __name__ == '__main__':
    feature_doc2vec()
    """
        加载数据和分词。。。
        加载数据和分词用时：16.51794695854187s
        抽取句子特征。。。
        抽取句子特种用时：2787.725296974182s
        持久化数据。。。
        持久化数据用时：2.6352829933166504s
        总用时：2806.8785269260406s
    """
