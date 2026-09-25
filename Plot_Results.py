import numpy as np
from matplotlib import pyplot as plt
from prettytable import PrettyTable
from sklearn.metrics import roc_curve, roc_auc_score
from itertools import cycle
import warnings
from sklearn import metrics

warnings.filterwarnings('ignore')

No_of_Dataset = 3


def Statistical(data):
    Min = np.min(data)
    Max = np.max(data)
    Mean = np.mean(data)
    Median = np.median(data)
    Std = np.std(data)
    return np.asarray([Min, Max, Mean, Median, Std])


def plotConvResults():
    # matplotlib.use('TkAgg')
    Fitness = np.load('Fitness.npy', allow_pickle=True)
    Algorithm = ['TERMS', 'PCOA-HR-CBA-AMVT-BGRU', 'LEA-HR-CBA-AMVT-BGRU', 'FSA-HR-CBA-AMVT-BGRU',
                 'OOA-HR-CBA-AMVT-BGRU', 'IPU-OOA-CBA-AMVT-BGRU']
    Terms = ['BEST', 'WORST', 'MEAN', 'MEDIAN', 'STD']
    for i in range(No_of_Dataset):
        Conv_Graph = np.zeros((len(Algorithm) - 1, len(Terms)))
        for j in range(len(Algorithm) - 1):  # for 5 algms
            Conv_Graph[j, :] = Statistical(Fitness[i, j, :])

        Table = PrettyTable()
        Table.add_column(Algorithm[0], Terms)
        for j in range(len(Algorithm) - 1):
            Table.add_column(Algorithm[j + 1], Conv_Graph[j, :])
        print('-------------------------------------------------- Statistical Analysis  ',
              '--------------------------------------------------')
        print(Table)

        length = np.arange(Fitness.shape[2])
        fig = plt.figure()
        fig.canvas.manager.set_window_title('Dataset-' + str(i + 1) + ' Convergence Curve')
        Conv_Graph = Fitness[i]
        plt.plot(length, Conv_Graph[0, :], color='r', linewidth=3, marker='*', markerfacecolor='red',
                 markersize=12, label=Algorithm[1])
        plt.plot(length, Conv_Graph[1, :], color='g', linewidth=3, marker='*', markerfacecolor='green',
                 markersize=12, label=Algorithm[2])
        plt.plot(length, Conv_Graph[2, :], color='b', linewidth=3, marker='*', markerfacecolor='blue',
                 markersize=12, label=Algorithm[3])
        plt.plot(length, Conv_Graph[3, :], color='m', linewidth=3, marker='*', markerfacecolor='magenta',
                 markersize=12, label=Algorithm[4])
        plt.plot(length, Conv_Graph[4, :], color='k', linewidth=3, marker='*', markerfacecolor='black',
                 markersize=12, label=Algorithm[5])
        plt.xlabel('No. of Iteration', fontname="Arial", fontsize=12, fontweight='bold', color='k')
        plt.ylabel('Cost Function', fontname="Arial", fontsize=12, fontweight='bold', color='k')
        plt.legend(loc=1, prop={'weight': 'bold', 'size': 12})
        plt.savefig("./Results/Conv_%s.png" % (i + 1))
        plt.show()


def Plot_ROC_Curve():
    lw = 3
    cls = ['GZS-ConvNet', 'ResNet-50', 'CNN-LSTM', ' CBA-AMVT-BGRU', 'IPU-OOA-CBA-AMVT-BGRU']
    for a in range(No_of_Dataset):  # For 3 Datasets
        Actual = np.load('Target_' + str(a + 1) + '.npy', allow_pickle=True)
        lenper = round(Actual.shape[0] * 0.75)
        Actual = Actual[lenper:, :]
        fig = plt.figure()
        fig.canvas.manager.set_window_title('Dataset-' + str(a + 1) + ' ROC Curve')
        colors = cycle(["blue", "darkorange", "limegreen", "deeppink", "black"])
        for i, color in zip(range(5), colors):  # For all classifiers
            Predicted = np.load('Y_Score_' + str(a + 1) + '.npy', allow_pickle=True)[i]
            false_positive_rate, true_positive_rate, _ = roc_curve(Actual.ravel(), Predicted.ravel())
            roc_auc = roc_auc_score(Actual.ravel(), Predicted.ravel())
            roc_auc = roc_auc * 100

            plt.plot(
                false_positive_rate,
                true_positive_rate,
                color=color,
                lw=2,
                label=f'{cls[i]} (AUC = {roc_auc:.2f} %)')

        plt.plot([0, 1], [0, 1], "k--", lw=lw)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel("False Positive Rate", fontname="Arial", fontsize=14, fontweight='bold', color='k')
        plt.ylabel("True Positive Rate", fontname="Arial", fontsize=14, fontweight='bold', color='k')
        plt.xticks(fontname="Arial", fontsize=14, fontweight='bold', color='#1d3557')
        plt.yticks(fontname="Arial", fontsize=14, fontweight='bold', color='#1d3557')
        plt.title("ROC Curve")
        plt.legend(loc="lower right", prop={'weight': 'bold', 'size': 12})
        path = "./Results/Dataset_%s_ROC.png" % (a + 1)
        plt.savefig(path)
        plt.show()


def plot_Alg_Results():
    eval = np.load('Evaluate_all.npy', allow_pickle=True)
    Terms = ['Accuracy', 'Sensitivity', 'Specificity', 'Precision', 'FPR', 'FNR', 'NPV', 'FDR', 'F1 Score',
             'MCC', 'FOR', 'PT', 'CSI', 'BA', 'FM', 'BM', 'MK', 'LR+', 'LR-', 'DOR', 'Prevalence']
    Graph_Terms = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18]
    bar_width = 0.15
    Algorithm = ['PCOA-HR-CBA-AMVT-BGRU', 'LEA-HR-CBA-AMVT-BGRU', 'FSA-HR-CBA-AMVT-BGRU', 'OOA-HR-CBA-AMVT-BGRU',
                 'IPU-OOA-CBA-AMVT-BGRU']
    Learning_Percentage = [35, 45, 55, 65, 75]
    for i in range(eval.shape[0]):
        for j in range(len(Graph_Terms)):
            Graph = np.zeros(eval.shape[1:3])
            for k in range(eval.shape[1]):
                for l in range(eval.shape[2]):
                    Graph[k, l] = eval[i, k, l, Graph_Terms[j] + 4]

            fig = plt.figure(figsize=(10, 7))
            ax = fig.add_axes([0.12, 0.12, 0.8, 0.8])
            fig.canvas.manager.set_window_title(
                'Dataset -' + str(i + 1) + ' Algorithm Comparison of Learning Percentage')
            X = np.arange(len(Learning_Percentage))
            plt.bar(X + 0.00, Graph[:, 0], color='yellowgreen', edgecolor='w', width=0.15, label=Algorithm[0])
            plt.bar(X + 0.15, Graph[:, 1], color='gold', edgecolor='w', width=0.15, label=Algorithm[1])
            plt.bar(X + 0.30, Graph[:, 2], color='mediumpurple', edgecolor='w', width=0.15, label=Algorithm[2])
            plt.bar(X + 0.45, Graph[:, 3], color='sandybrown', edgecolor='w', width=0.15, label=Algorithm[3])
            plt.bar(X + 0.60, Graph[:, 4], color='k', edgecolor='w', width=0.15, label=Algorithm[4])
            plt.xticks(X + bar_width * 2, ['35', '45', '55', '65', '75'], fontname="Arial", fontsize=14,
                       fontweight='bold', color='k')
            plt.xlabel('Learning Percentage', fontname="Arial", fontsize=14, fontweight='bold', color='k')
            plt.ylabel(Terms[Graph_Terms[j]], fontname="Arial", fontsize=14, fontweight='bold', color='k')
            plt.yticks(fontname="Arial", fontsize=14, fontweight='bold', color='#35530a')
            dot_markers = [plt.Line2D([2], [2], marker='s', color='w', markerfacecolor=color, markersize=10) for color
                           in ['yellowgreen', 'gold', 'mediumpurple', 'sandybrown', 'k']]
            plt.legend(dot_markers, Algorithm, loc='upper center', bbox_to_anchor=(0.48, 1.08), fontsize=10,
                       frameon=False, ncol=3, prop={'weight': 'bold', 'size': 10})
            plt.gca().spines['top'].set_visible(False)
            plt.gca().spines['right'].set_visible(False)
            plt.gca().spines['left'].set_visible(False)
            plt.gca().spines['bottom'].set_visible(True)
            plt.grid(axis='y', linestyle='--', alpha=0.7)
            plt.tight_layout()
            path = "./Results/Dataset_%s_%s_Alg_bar.png" % (i + 1, Terms[Graph_Terms[j]])
            plt.savefig(path)
            plt.show()


def Plot_Mod_Results():
    eval = np.load('Evaluate_all.npy', allow_pickle=True)
    Terms = ['Accuracy', 'Sensitivity', 'Specificity', 'Precision', 'FPR', 'FNR', 'NPV', 'FDR', 'F1 Score',
             'MCC', 'FOR', 'PT', 'CSI', 'BA', 'FM', 'BM', 'MK', 'LR+', 'LR-', 'DOR', 'Prevalence']
    Graph_Terms = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18]
    bar_width = 0.15
    Classifier = ['GZS-ConvNet', 'ResNet-50', 'CNN-LSTM', ' CBA-AMVT-BGRU', 'IPU-OOA-CBA-AMVT-BGRU']
    Learning_Percentage = [35, 45, 55, 65, 75]
    for i in range(eval.shape[0]):
        for j in range(len(Graph_Terms)):
            Graph = np.zeros(eval.shape[1:3])
            for k in range(eval.shape[1]):
                for l in range(eval.shape[2]):
                    Graph[k, l] = eval[i, k, l, Graph_Terms[j] + 4]

            fig = plt.figure()
            ax = fig.add_axes([0.12, 0.12, 0.8, 0.8])
            fig.canvas.manager.set_window_title('Dataset -' + str(i + 1) + ' Method Comparison of Learning Percentage')
            X = np.arange(len(Learning_Percentage))
            plt.bar(X + 0.00, Graph[:, 5], color='#9e0059', edgecolor='w', width=0.15, label=Classifier[0])
            plt.bar(X + 0.15, Graph[:, 6], color='#390099', edgecolor='w', width=0.15, label=Classifier[1])
            plt.bar(X + 0.30, Graph[:, 7], color='#ff5400', edgecolor='w', width=0.15, label=Classifier[2])
            plt.bar(X + 0.45, Graph[:, 8], color='#38a3a5', edgecolor='w', width=0.15, label=Classifier[3])
            plt.bar(X + 0.60, Graph[:, 4], color='k', edgecolor='w', width=0.15, label=Classifier[4])
            plt.xticks(X + bar_width * 2, ['35', '45', '55', '65', '75'], fontname="Arial", fontsize=14,
                       fontweight='bold', color='k')
            plt.xlabel('Learning Percentage', fontname="Arial", fontsize=14, fontweight='bold', color='k')
            plt.ylabel(Terms[Graph_Terms[j]], fontname="Arial", fontsize=14, fontweight='bold', color='k')
            plt.yticks(fontname="Arial", fontsize=14, fontweight='bold', color='#35530a')
            dot_markers = [plt.Line2D([2], [2], marker='s', color='w', markerfacecolor=color, markersize=10) for color
                           in ['#9e0059', '#390099', '#ff5400', '#38a3a5', 'k']]
            plt.legend(dot_markers, Classifier, loc='upper center', bbox_to_anchor=(0.5, 1.10), fontsize=10,
                       frameon=False, ncol=3, prop={'weight': 'bold', 'size': 10})
            plt.gca().spines['top'].set_visible(False)
            plt.gca().spines['right'].set_visible(False)
            plt.gca().spines['left'].set_visible(False)
            plt.gca().spines['bottom'].set_visible(True)
            plt.grid(axis='y', linestyle='--', alpha=0.7)
            plt.tight_layout()
            path = "./Results/Dataset_%s_%s_Mod_bar.png" % (i + 1, Terms[Graph_Terms[j]])
            plt.savefig(path)
            plt.show()


def Table():
    eval = np.load('Evaluate.npy', allow_pickle=True)
    Algorithm = ['Kfold', 'PCOA-HR-CBA-AMVT-BGRU', 'LEA-HR-CBA-AMVT-BGRU', 'FSA-HR-CBA-AMVT-BGRU',
                 'OOA-HR-CBA-AMVT-BGRU', 'IPU-OOA-CBA-AMVT-BGRU']
    Classifier = ['Kfold', 'GZS-ConvNet', 'ResNet-50', 'CNN-LSTM', ' CBA-AMVT-BGRU', 'IPU-OOA-CBA-AMVT-BGRU']
    Terms = ['Accuracy', 'Sensitivity', 'Specificity', 'Precision', 'FPR', 'FNR', 'NPV', 'FDR', 'F1 Score',
             'MCC', 'FOR', 'PT', 'CSI', 'BA', 'FM', 'BM', 'MK', 'LR+', 'LR-', 'DOR', 'Prevalence']
    Graph_Terms = np.array([0, 2, 5, 8, 12, 15, 16]).astype(int)
    Table_Terms = [0, 2, 5, 8, 12, 15, 16]
    table_terms = [Terms[i] for i in Table_Terms]
    Kfold = ['Kfold 1', 'Kfold 2', 'Kfold 3', 'Kfold 4', 'Kfold 5']
    for i in range(eval.shape[0]):
        for k in range(len(Table_Terms)):
            value = eval[i, :, :, 4:]

            Table = PrettyTable()
            Table.add_column(Algorithm[0], Kfold)
            for j in range(len(Algorithm) - 1):
                Table.add_column(Algorithm[j + 1], value[:, j, Graph_Terms[k]])
            print('------------------------------- Dataset- ', i + 1, table_terms[k], '  Algorithm Comparison',
                  '---------------------------------------')
            print(Table)

            Table = PrettyTable()
            Table.add_column(Classifier[0], Kfold)
            for j in range(len(Classifier) - 1):
                Table.add_column(Classifier[j + 1], value[:, len(Algorithm) + j - 1, Graph_Terms[k]])
            print('------------------------------- Dataset- ', i + 1, table_terms[k], '  Classifier Comparison',
                  '---------------------------------------')
            print(Table)


def Proposed_Plots_Results():
    eval = np.load('Eval_all.npy', allow_pickle=True)
    Terms = ['Accuracy', 'Sensitivity', 'Specificity', 'Precision', 'FPR', 'FNR', 'NPV', 'FDR', 'F1 Score',
             'MCC', 'FOR', 'PT', 'CSI', 'BA', 'FM', 'BM', 'MK', 'LR+', 'LR-', 'DOR', 'Prevalence']
    Graph_Terms = [0, 5, 9, 10, 12, 13]
    Act_Fun = ['Linear', 'ReLu', 'Softmax', 'Sigmoid', 'Tanh']
    Algorithm = ['OOA-HR-CBA-AMVT-BGRU', 'IPU-OOA-CBA-AMVT-BGRU']
    Classifier = ['CBA-AMVT-BGRU', 'IPU-OOA-CBA-AMVT-BGRU']
    for i in range(eval.shape[0]):
        for j in range(len(Graph_Terms)):
            Graph = np.zeros(eval.shape[1:3])
            for k in range(eval.shape[1]):
                for l in range(eval.shape[2]):
                    Graph[k, l] = eval[i, k, l, Graph_Terms[j] + 4]

            fig = plt.figure()
            ax = fig.add_axes([0.12, 0.12, 0.8, 0.8])
            ax.yaxis.grid()
            fig.canvas.manager.set_window_title(
                'Dataset - ' + str(i + 1) + 'Algorithm Comparison of Activation Function')
            X = np.arange(len(Act_Fun) - 2)
            plt.bar(X + 0.00, Graph[:3, 3], color='#5c8001', linewidth=2, width=0.40,
                    label=Algorithm[0])
            plt.bar(X + 0.40, Graph[:3, 4], color='k', linewidth=2, width=0.40,
                    label=Algorithm[1])
            plt.xticks(X + 0.20, ['Linear', 'ReLu', 'Softmax'], fontsize=12,
                       fontname="Arial",
                       fontweight='bold', color='k')
            plt.ylabel(Terms[Graph_Terms[j]], fontsize=12, fontname="Arial", fontweight='bold', color='k')
            plt.xlabel('Activation Function', fontsize=12, fontname="Arial", fontweight='bold', color='k')
            plt.yticks(fontname="Arial", fontsize=12, fontweight='bold', color='#35530a')
            plt.gca().spines['top'].set_visible(False)
            plt.gca().spines['right'].set_visible(False)
            plt.gca().spines['left'].set_visible(False)
            plt.gca().spines['bottom'].set_visible(False)
            dot_markers = [plt.Line2D([2], [2], marker='o', color='w', markerfacecolor=color, markersize=10) for color
                           in ['#5c8001', 'k']]
            plt.legend(dot_markers, Algorithm, loc='upper center', bbox_to_anchor=(0.5, 1.10), fontsize=12,
                       frameon=False, ncol=len(Algorithm), prop={'weight': 'bold', 'size': 12})
            plt.tight_layout()
            path = "./Results/Dataset_%s_%s_Alg_Proposed__bar.png" % (i + 1, Terms[Graph_Terms[j]])
            plt.savefig(path)
            plt.show()

            fig = plt.figure()
            ax = fig.add_axes([0.12, 0.12, 0.8, 0.8])
            ax.yaxis.grid()
            fig.canvas.manager.set_window_title('Dataset - ' + str(i + 1) + 'Method Comparison of Activation Function')
            X = np.arange(len(Act_Fun) - 2)
            plt.bar(X + 0.00, Graph[:3, 8], color='#ff5a5f', width=0.40, label=Classifier[0])
            plt.bar(X + 0.40, Graph[:3, 4], color='k', width=0.40, label=Classifier[1])
            plt.xticks(X + 0.20, ['Linear', 'ReLu', 'Softmax'], fontname="Arial",
                       fontsize=12,
                       fontweight='bold', color='k')
            plt.ylabel(Terms[Graph_Terms[j]], fontname="Arial", fontsize=12, fontweight='bold', color='k')
            plt.yticks(fontname="Arial", fontsize=12, fontweight='bold', color='#35530a')
            plt.xlabel('Activation Function', fontsize=12, fontname="Arial", fontweight='bold', color='k')
            plt.gca().spines['top'].set_visible(False)
            plt.gca().spines['right'].set_visible(False)
            plt.gca().spines['left'].set_visible(False)
            plt.gca().spines['bottom'].set_visible(False)
            dot_markers = [plt.Line2D([2], [2], marker='o', color='w', markerfacecolor=color, markersize=10) for color
                           in ['#ff5a5f', 'k']]
            plt.legend(dot_markers, Classifier, loc='upper center', bbox_to_anchor=(0.5, 1.10), fontsize=12,
                       frameon=False, ncol=len(Classifier), prop={'weight': 'bold', 'size': 12})
            plt.tight_layout()
            path = "./Results/Dataset_%s_%s_mod_Proposed_bar.png" % (i + 1, Terms[Graph_Terms[j]])
            plt.savefig(path)
            plt.show()


def Reg_Plot_Results():
    eval = np.load('Evaluate_Seg_all.npy', allow_pickle=True)
    Terms = ['Accuarcy', 'Sensitivity', 'Precision', 'F1 Score', 'FAR', 'FRR', 'GAR']
    Algorithm = ['PCOA-AMFN', 'LEA-AMFN ', 'FSA-AMFN', 'OOA-AMFN', 'IPU-OOA-AMFN']
    Method = ['CNN', 'Resnet', 'Vgg-16', ' MFN', 'IPU-OOA-AMFN']
    Graph_Terms = [0, 1, 2, 3, 4, 5, 6]
    Epochs = [20, 40, 60, 80, 100]
    for i in range(eval.shape[0]):
        Graph = np.zeros((eval.shape[1], eval.shape[2]))
        for j in range(len(Graph_Terms)):
            for k in range(eval.shape[1]):
                for l in range(eval.shape[2]):
                    Graph[k, l] = eval[i, k, l, Graph_Terms[j] + 4]
            fig = plt.figure()
            ax = fig.add_axes([0.12, 0.12, 0.7, 0.7])
            fig.canvas.manager.set_window_title(
                'Dataset - ' + str(i + 1) + ' No. of Epochs vs ' + Terms[Graph_Terms[j]])
            plt.plot(Epochs, Graph[:, 0], color='#d62828', linewidth=3, marker='o', markersize=12,
                     label=Algorithm[0])
            plt.plot(Epochs, Graph[:, 1], color='#00b4d8', linewidth=3, marker='o', markersize=12,
                     label=Algorithm[1])
            plt.plot(Epochs, Graph[:, 2], color='#f15bb5', linewidth=3, marker='o', markersize=12,
                     label=Algorithm[2])
            plt.plot(Epochs, Graph[:, 3], color='#4f772d', linewidth=3, marker='o', markersize=12,
                     label=Algorithm[3])
            plt.plot(Epochs, Graph[:, 4], color='k', linewidth=3, marker='o', markersize=12,
                     label=Algorithm[4])
            plt.gca().spines['top'].set_visible(False)
            plt.gca().spines['right'].set_visible(False)
            plt.gca().spines['left'].set_visible(False)
            plt.gca().spines['bottom'].set_visible(False)
            dot_markers = [plt.Line2D([2], [2], marker='o', color='w', markerfacecolor=color, markersize=13) for color
                           in ['#d62828', '#00b4d8', '#f15bb5', '#4f772d', 'k']]
            plt.legend(dot_markers, Algorithm, loc='upper center', bbox_to_anchor=(0.5, 1.22), fontsize=10,
                       frameon=False, ncol=3, prop={'weight': 'bold', 'size': 12})
            plt.tight_layout()
            plt.xticks(Epochs, ('20', '40', '60', '80', '100'), fontname="Arial", fontsize=14,
                       fontweight='bold',
                       color='#35530a')
            plt.yticks(fontname="Arial", fontsize=14, fontweight='bold',
                       color='#35530a')
            plt.xlabel('No. of Epochs', fontname="Arial", fontsize=14, fontweight='bold', color='k')
            plt.ylabel(Terms[Graph_Terms[j]], fontname="Arial", fontsize=14, fontweight='bold', color='k')
            path = "./Results/Dataset_%s_%s_Recogn_Alg_line.png" % (i + 1, Terms[Graph_Terms[j]])
            plt.savefig(path)
            plt.show()

            fig = plt.figure()
            ax = fig.add_axes([0.12, 0.12, 0.7, 0.7])
            fig.canvas.manager.set_window_title(
                'Dataset - ' + str(i + 1) + ' No. of Epochs vs ' + Terms[Graph_Terms[j]])
            plt.plot(Epochs, Graph[:, 5], color='#a600cc', linewidth=3, marker='o', markersize=12,
                     label=Method[0])
            plt.plot(Epochs, Graph[:, 6], color='#f77f00', linewidth=3, marker='o', markersize=12,
                     label=Method[1])
            plt.plot(Epochs, Graph[:, 7], color='r', linewidth=3, marker='o', markersize=12,
                     label=Method[2])
            plt.plot(Epochs, Graph[:, 8], color='b', linewidth=3, marker='o', markersize=12,
                     label=Method[3])
            plt.plot(Epochs, Graph[:, 4], color='k', linewidth=3, marker='o', markersize=12,
                     label=Method[4])
            plt.gca().spines['top'].set_visible(False)
            plt.gca().spines['right'].set_visible(False)
            plt.gca().spines['left'].set_visible(False)
            plt.gca().spines['bottom'].set_visible(False)
            dot_markers = [plt.Line2D([2], [2], marker='o', color='w', markerfacecolor=color, markersize=13) for color
                           in ['#a600cc', '#f77f00', 'r', 'b', 'k']]
            plt.legend(dot_markers, Method, loc='upper center', bbox_to_anchor=(0.5, 1.22), fontsize=10,
                       frameon=False, ncol=3, prop={'weight': 'bold', 'size': 12})
            plt.tight_layout()
            plt.xticks(Epochs, ('20', '40', '60', '80', '100'), fontname="Arial", fontsize=14,
                       fontweight='bold',
                       color='#35530a')
            plt.yticks(fontname="Arial", fontsize=14, fontweight='bold',
                       color='#35530a')
            plt.xlabel('No. of Epochs', fontname="Arial", fontsize=14, fontweight='bold', color='k')
            plt.ylabel(Terms[Graph_Terms[j]], fontname="Arial", fontsize=14, fontweight='bold', color='k')
            path = "./Results/Dataset_%s_%s_Recogn_Mod_line.png" % (i + 1, Terms[Graph_Terms[j]])
            plt.savefig(path)
            plt.show()


def Reg_Table():
    eval = np.load('Evaluate_Seg.npy', allow_pickle=True)
    Algorithm = ['BatchSize', 'PCOA-AMFN', 'LEA-AMFN ', 'FSA-AMFN', 'OOA-AMFN', 'IPU-OOA-AMFN']
    Classifier = ['BatchSize', 'CNN', 'Resnet', 'Vgg-16', ' MFN', 'IPU-OOA-AMFN']
    Terms = ['Accuarcy', 'Sensitivity', 'Precision', 'F1 Score', 'FAR', 'FRR', 'GAR']
    Graph_Terms = np.array([0, 4, 5, 6]).astype(int)
    Table_Terms = [0, 4, 5, 6]
    table_terms = [Terms[i] for i in Table_Terms]
    BatchSize = ['4', '16', '32', '48', '64']
    for i in range(eval.shape[0]):
        for k in range(len(Table_Terms)):
            value = eval[i, :, :, 4:]

            Table = PrettyTable()
            Table.add_column(Algorithm[0], BatchSize)
            for j in range(len(Algorithm) - 1):
                Table.add_column(Algorithm[j + 1], value[:, j, Graph_Terms[k]])
            print('------------------------------- Dataset- ', i + 1, table_terms[k],
                  '  Algorithm Comparison for Recognition',
                  '---------------------------------------')
            print(Table)

            Table = PrettyTable()
            Table.add_column(Classifier[0], BatchSize)
            for j in range(len(Classifier) - 1):
                Table.add_column(Classifier[j + 1], value[:, len(Algorithm) + j - 1, Graph_Terms[k]])
            print('------------------------------- Dataset- ', i + 1, table_terms[k],
                  '  Classifier Comparison for Recognition',
                  '---------------------------------------')
            print(Table)


if __name__ == '__main__':
    # plotConvResults()
    # plot_Alg_Results()
    Plot_Mod_Results()
    Plot_ROC_Curve()
    Table()
    Proposed_Plots_Results()
    # Reg_Plot_Results()
    # Reg_Table()
