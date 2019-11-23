# -*- coding:utf-8 -*-.
__author__ = 'caixinjun'

import pandas as pd
import networkx as nx
from embedding import Embedding
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from tqdm import trange


def link_prediction(user_embeddings, item_embeddings):
    print("starting link prediction...")
    train = pd.read_csv('../data/Amazon/amazon_train.csv', header=None)
    test = pd.read_csv('../data/Amazon/amazon_train.csv', header=None)
    vectors = user_embeddings.append(item_embeddings)
    print("generating features...")
    for i in trange(0, 64):
        vec_dict = vectors.to_dict()[i]
        train['user_vec_%s' % (i)] = train[0].map(vec_dict)
        train['item_vec_%s' % (i)] = train[1].map(vec_dict)
        test['user_vec_%s' % (i)] = test[0].map(vec_dict)
        test['item_vec_%s' % (i)] = test[1].map(vec_dict)
    y_train = train.pop(2)
    y_test = test.pop(2)
    train.drop([0, 1], axis=1, inplace=True)
    test.drop([0, 1], axis=1, inplace=True)
    X_train = train.fillna(0.0)
    X_test = test.fillna(0.0)

    print("training....")
    lr = LogisticRegression()
    lr.fit(X_train, y_train)
    y_pred = lr.predict(X_test)

    fpr, tpr, thresholds = metrics.roc_curve(y_test, y_pred)
    average_precision = metrics.average_precision_score(y_test, y_pred)
    print("auc_roc: ", str(metrics.auc(fpr, tpr)))
    print("auc_pr: ", str(average_precision))



if __name__ == "__main__":

    G = nx.read_weighted_edgelist('../data/Amazon/amazon_bg.txt', create_using=nx.Graph(), nodetype=None)
    model = Embedding(G, walk_length=15, num_walks=80, workers=2)
    model.train(embed_size=64, window_size=5, iter=5, workers=2)

    user_embeddings, item_embeddings = model.get_embeddings()

    columns = list(map(lambda x: "X_" + str(x), range(64)))
    user_embeddings = pd.DataFrame(user_embeddings)
    user_embeddings = pd.DataFrame(user_embeddings.values.T, index=user_embeddings.columns, columns=user_embeddings.index)
    item_embeddings = pd.DataFrame(item_embeddings)
    item_embeddings = pd.DataFrame(item_embeddings.values.T, index=item_embeddings.columns, columns=item_embeddings.index)

    user_embeddings.to_csv('../output/user_embeddings.csv')
    item_embeddings.to_csv('../output/item_embeddings.csv')
    link_prediction(user_embeddings, item_embeddings)