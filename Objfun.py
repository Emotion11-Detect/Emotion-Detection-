import numpy as np

from Evaluation import Reg_evaluation, evaluation
from Global_Vars import Global_Vars
from MobileFaceNet import Model_MobileFaceNet
from Model_Ensemble import Model_Ensemble


def objfun_1(Soln):
    data = Global_Vars.Data
    Tar = Global_Vars.Target
    Fitn = np.zeros(Soln.shape[0])
    dimension = len(Soln.shape)
    if dimension == 2:
        learnper = round(data.shape[0] * 0.75)
        for i in range(Soln.shape[0]):
            sol = np.round(Soln[i, :]).astype(np.int16)
            Train_Data = data[:learnper, :]
            Train_Target = Tar[:learnper, :]
            Test_Data = data[learnper:, :]
            Test_Target = Tar[learnper:, :]
            Eval, pred = Model_MobileFaceNet(Train_Data, Train_Target, Test_Data, Test_Target, sol)
            Eval = Reg_evaluation(Test_Target, pred)
            Fitn[i] = 1 / Eval[4]
        return Fitn
    else:
        learnper = round(data.shape[0] * 0.75)
        sol = np.round(Soln).astype(np.int16)
        Train_Data = data[:learnper, :]
        Train_Target = Tar[:learnper, :]
        Test_Data = data[learnper:, :]
        Test_Target = Tar[learnper:, :]
        Eval, pred = Model_MobileFaceNet(Train_Data, Train_Target, Test_Data, Test_Target, sol)
        Eval = Reg_evaluation(Test_Target, pred)
        Fitn = 1 / Eval[4]
        return Fitn


def objfun_2(Soln):
    data = Global_Vars.Data
    Tar = Global_Vars.Target
    Fitn = np.zeros(Soln.shape[0])
    dimension = len(Soln.shape)
    if dimension == 2:
        learnper = round(data.shape[0] * 0.75)
        for i in range(Soln.shape[0]):
            sol = np.round(Soln[i, :]).astype(np.int16)
            Train_Data = data[:learnper, :]
            Train_Target = Tar[:learnper, :]
            Test_Data = data[learnper:, :]
            Test_Target = Tar[learnper:, :]
            Eval, pred = Model_Ensemble(Train_Data, Train_Target, Test_Data, Test_Target, sol)
            Eval = evaluation(Test_Target, pred)
            Fitn[i] = (1 / (Eval[4] + Eval[13])) + Eval[9]
        return Fitn
    else:
        learnper = round(data.shape[0] * 0.75)
        sol = np.round(Soln).astype(np.int16)
        Train_Data = data[:learnper, :]
        Train_Target = Tar[:learnper, :]
        Test_Data = data[learnper:, :]
        Test_Target = Tar[learnper:, :]
        Eval, pred = Model_Ensemble(Train_Data, Train_Target, Test_Data, Test_Target, sol)
        Eval = evaluation(Test_Target, pred)
        Fitn = (1 / (Eval[4] + Eval[13])) + Eval[9]
        return Fitn