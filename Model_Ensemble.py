import numpy as np
from Evaluation import evaluation
from Model_Bi_GRU import Model_Bi_GRU
from Model_CBAM import Model_CBAM
from Model_MViT import Model_MViT


def HighRanking(Pred):
    Pred = np.asarray(Pred)
    pred = np.zeros((Pred.shape[1], 1))
    for i in range(Pred.shape[1]):
        p = Pred[:, i]
        uniq, count = np.unique(p, return_counts=True)
        index = np.argmax(count)
        pred[i] = uniq[index]

    return pred


def Model_Ensemble(Train_Data, Train_Target, Test_Data, Test_Target, activation, sol=None):
    if sol is None:
        sol = [5, 5, 0.01]
    eval, Pred_MViT = Model_MViT(Train_Data, Train_Target, Test_Data, Test_Target, activation, sol)
    eval, Pred_GRU = Model_Bi_GRU(Train_Data, Train_Target, Test_Data, Test_Target, activation)
    eval, Pred_CBAM = Model_CBAM(Train_Data, Train_Target, Test_Data, Test_Target, activation)

    pred = [np.reshape(Pred_MViT, (-1, 1)), np.reshape(Pred_GRU, (-1, 1)), np.reshape(Pred_CBAM, (-1, 1))]
    predict = HighRanking(pred)
    eval = evaluation(Test_Target, predict)

    return eval, predict

