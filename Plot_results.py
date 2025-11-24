from itertools import cycle
import matplotlib.pyplot as plt
import numpy as np
from prettytable import PrettyTable
from sklearn.metrics import roc_curve, confusion_matrix
from sklearn import metrics

No_of_Dataset = 1


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
    Algorithm = ['TERMS', 'LEA-ARHN', 'FDA-ARHN', 'AOA-ARHN', 'FLO-ARHN', 'MRV-FLO-ARHN']
    Terms = ['BEST', 'WORST', 'MEAN', 'MEDIAN', 'STD']

    for i in range(Fitness.shape[0]):
        Conv_Graph = np.zeros((len(Algorithm) - 1, 5))
        for j in range(len(Algorithm) - 1):  # for 5 algms
            Conv_Graph[j, :] = Statistical(Fitness[i, j, :])

        Table = PrettyTable()
        Table.add_column(Algorithm[0], Terms)
        for j in range(len(Algorithm) - 1):
            Table.add_column(Algorithm[j + 1], Conv_Graph[j, :])
        print('--------------------------------------------------'' Statistical Report '
              '--------------------------------------------------')
        print(Table)

        Conv_Graph = Fitness[i]
        length = np.arange((Conv_Graph.shape[1]))
        plt.plot(length, Conv_Graph[0, :], color='r', linewidth=3, marker='*', markerfacecolor='red',
                 markersize=12, label='LEA-ARHN')
        plt.plot(length, Conv_Graph[1, :], color=[0, 0.5, 0.5], linewidth=3, marker='*', markerfacecolor='green',
                 markersize=12, label='FDA-ARHN')
        plt.plot(length, Conv_Graph[2, :], color=[0.5, 0, 0.5], linewidth=3, marker='*', markerfacecolor='blue',
                 markersize=12, label='AOA-ARHN')
        plt.plot(length, Conv_Graph[3, :], color='m', linewidth=3, marker='*', markerfacecolor='magenta',
                 markersize=12, label='FLO-ARHN')
        plt.plot(length, Conv_Graph[4, :], color='k', linewidth=3, marker='*', markerfacecolor='black',
                 markersize=12, label='MRV-FLO-ARHN')
        plt.xlabel('Iteration')
        plt.ylabel('Cost Function')
        plt.legend(loc=1)
        plt.savefig("./Results/Conv.png")
        plt.show()


def Plot_Results_Learn():
    eval = np.load('Eval_all_Learning_Percent.npy', allow_pickle=True)
    Terms = ['MD', 'SMAPE', 'MASE', 'MAE', 'RMSE', 'MSE', 'ONENORM', 'TWONORM', 'Infinity Norm', 'Accuracy']
    Graph_Term = [2, 3, 4, 5, 9]
    Learn = [0.15, 0.30, 0.45, 0.60, 0.75]
    for j in range(len(Graph_Term)):
        Graph = np.zeros((eval.shape[0], eval.shape[1]))
        for k in range(eval.shape[0]):
            for l in range(eval.shape[1]):
                Graph[k, l] = eval[k, l, Graph_Term[j]]

        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_axes([0.12, 0.12, 0.7, 0.7])
        ax.set_facecolor("#f0f0f0")
        fig.canvas.manager.set_window_title('Learning Percentage vs ' + Terms[Graph_Term[j]])
        plt.plot(Learn, Graph[:, 0], color='#ffbe0b', linewidth=3, marker='o', markersize=10, label="LEA-ARHN")
        plt.plot(Learn, Graph[:, 1], color='#fb5607', linewidth=3, marker='o', markersize=10, label="FDA-ARHN")
        plt.plot(Learn, Graph[:, 2], color='#ff006e', linewidth=3, marker='o', markersize=10, label="AOA-ARHN")
        plt.plot(Learn, Graph[:, 3], color='#8338ec', linewidth=3, marker='o', markersize=10, label="FLO-ARHN")
        plt.plot(Learn, Graph[:, 4], color='k', linewidth=3, markersize=16, label="MRV-FLO-ARHN")
        plt.gca().spines['top'].set_visible(False)
        plt.gca().spines['right'].set_visible(False)
        plt.gca().spines['left'].set_visible(False)
        plt.gca().spines['bottom'].set_visible(False)
        plt.legend(loc='upper center', bbox_to_anchor=(0.80, 0.97), ncol=1, fancybox=True, shadow=False, framealpha=0.5)

        plt.xticks(Learn, ('35', '45', '55', '65', '75'), fontname="Arial", fontsize=12, fontweight='bold',
                   color='#35530a')
        plt.yticks(fontname="Arial", fontsize=12, fontweight='bold',
                   color='#35530a')
        plt.xlabel('Learning Percentage (%)', fontname="Arial", fontsize=12, fontweight='bold', color='k')
        plt.ylabel(Terms[Graph_Term[j]], fontname="Arial", fontsize=12, fontweight='bold', color='k')
        plt.grid(color='#aec3b0', linestyle='-', linewidth=2)
        path = "./Results/Learn_Per_%s_Alg_line.png" % (Terms[Graph_Term[j]])
        plt.savefig(path)
        plt.show()

        # ----------------------------------------------------------------------------------------------

        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_axes([0.12, 0.12, 0.7, 0.7])
        ax.set_facecolor("#f0f0f0")
        fig.canvas.manager.set_window_title('Learning Percentage vs ' + Terms[Graph_Term[j]])
        plt.plot(Learn, Graph[:, 5], color='#ef476f', linewidth=3, marker='o', markersize=12, label="DSVM")
        plt.plot(Learn, Graph[:, 6], color='#ffd166', linewidth=3, marker='o', markersize=12, label="DCNN")
        plt.plot(Learn, Graph[:, 7], color='#06d6a0', linewidth=3, marker='o', markersize=12, label="LSTM")
        plt.plot(Learn, Graph[:, 8], color='#118ab2', linewidth=3, marker='o', markersize=12, label="RHN")
        plt.plot(Learn, Graph[:, 9], color='k', linewidth=3, markersize=12, label="MRV-FLO-ARHN")
        plt.gca().spines['top'].set_visible(False)
        plt.gca().spines['right'].set_visible(False)
        plt.gca().spines['left'].set_visible(False)
        plt.gca().spines['bottom'].set_visible(False)
        plt.legend(loc='upper center', bbox_to_anchor=(0.80, 0.97), ncol=1, fancybox=True, shadow=False, framealpha=0.5)

        plt.xticks(Learn, ('35', '45', '55', '65', '75'), fontname="Arial", fontsize=12, fontweight='bold',
                   color='#35530a')
        plt.yticks(fontname="Arial", fontsize=12, fontweight='bold',
                   color='#35530a')
        plt.xlabel('Learning Percentage (%)', fontname="Arial", fontsize=12, fontweight='bold', color='k')
        plt.ylabel(Terms[Graph_Term[j]], fontname="Arial", fontsize=12, fontweight='bold', color='k')
        plt.grid(color='#aec3b0', linestyle='-', linewidth=2)
        path = "./Results/Learn_Per_%s_Mtd_line.png" % (Terms[Graph_Term[j]])
        plt.savefig(path)
        plt.show()


def Plot_Results_Epoch():  # Table Results
    # matplotlib.use('TkAgg')
    eval = np.load('Eval_all_Epoch.npy', allow_pickle=True)
    Terms = ['MD', 'SMAPE', 'MASE', 'MAE', 'RMSE', 'MSE', 'ONENORM', 'TWONORM', 'Infinity Norm', 'Accuracy']

    Algorithm = ['Epochs\Algorithm', 'LEA-ARHN', 'FDA-ARHN', 'AOA-ARHN', 'FLO-ARHN', 'MRV-FLO-ARHN']
    Classifier = ['Epochs\Methods ', 'DSVM', 'DCNN', 'LSTM', 'RHN', 'MRV-FLO-ARHN']
    Epoch = ['50', '100', '150', '200', '250']

    Graph_Term = [2, 3, 4, 5, 9]

    for i in range(eval.shape[0]):
        # for k in range(eval.shape[1] + 2):
        for k in range(len(Graph_Term)):
            value = eval[i, :, :, Graph_Term[k]]
            Table = PrettyTable()
            Table.add_column(Algorithm[0], Epoch[:])
            for j in range(len(Algorithm) - 1):
                Table.add_column(Algorithm[j + 1], value[:, j])

            print('--------------------------------------------------' + Terms[Graph_Term[k]],
                  '-Algorithm Comparison ',
                  '--------------------------------------------------')
            print(Table)

            Table = PrettyTable()
            Table.add_column(Classifier[0], Epoch[:])
            for j in range(len(Classifier) - 1):
                Table.add_column(Classifier[j + 1], value[:, j + 5])
            print('--------------------------------------------------' + Terms[Graph_Term[k]],
                  '-Classifier Comparison',
                  '--------------------------------------------------')
            print(Table)



plotConvResults()
Plot_Results_Learn()
Plot_Results_Epoch()

