import matplotlib.pyplot as plt
import numpy as np
from prettytable import PrettyTable


def Plot_Activation_function():
    Terms = ['MAE', 'MD', 'MASE', 'SMAPE', 'RMSE', 'ONE-NORM', 'TWO-NORM']
    Eval = np.load('Eval_pred.npy', allow_pickle=True)
    learn = [1, 2, 3, 4, 5]
    X = np.arange(5)
    Graph_Term = [0,1,2,3,4,5,6]
    width = 0.15  # Width of each bar in the histogram
    for j in range(len(Graph_Term)):
        Graph = np.zeros((Eval.shape[0], Eval.shape[1]))
        for k in range(Eval.shape[0]):
            for l in range(Eval.shape[1]):
                Graph[k, l] = Eval[k, l, Graph_Term[j]]

        X = np.arange(len(learn))  # The x locations for the groups

        plt.bar(X - 2 * width, Graph[:, 0], width, color='purple', label='POA-SRF-CAALSTM')
        plt.bar(X - width, Graph[:, 1], width, color='DeepPink', label='HCO-SRF-CAALSTM')
        plt.bar(X, Graph[:, 2], width, color='cyan', label='DO-SRF-CAALSTM')
        plt.bar(X + width, Graph[:, 3], width, color='red', label='OOA-SRF-CAALSTM')
        plt.bar(X + 2 * width, Graph[:, 4], width, color='k', label="IOOA-SRF-CAALSTM")

        plt.xlabel('Activation Function')
        plt.xticks(X, ('Linear', 'Tanh', 'Softmax', 'Relu', 'Sigmoid'))  # Set x-ticks to correspond to activations
        plt.ylabel(Terms[Graph_Term[j]])

        # Display legend
        plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15),
                   ncol=3, fancybox=True, shadow=True)

        # Save the plot as an image file
        path = "./Results/_%s_histogram.png" % (Terms[Graph_Term[j]])
        plt.savefig(path)
        plt.show()

    eval = np.load('Eval_pred.npy', allow_pickle=True)
    Terms = ['MAE', 'MD', 'MASE', 'SMAPE', 'RMSE', 'ONE-NORM', 'TWO-NORM']
    Graph_Term = [0,1,2,3,4,5,6]
    for j in range(len(Graph_Term)):
        Graph = np.zeros((eval.shape[0], eval.shape[1]))
        for k in range(eval.shape[0]):
            for l in range(eval.shape[1]):
                Graph[k, l] = Eval[k, l, j]
        fig = plt.figure()
        ax = fig.add_axes([0.12, 0.1, 0.7, 0.8])
        X = np.arange(5)
        ax.bar(X + 0.00, Graph[:, 5], color='#f97306', edgecolor='k', width=0.10, hatch="+", label="DNN")
        ax.bar(X + 0.10, Graph[:, 6], color='#f10c45', edgecolor='k', width=0.10, hatch="+", label="DBN")
        ax.bar(X + 0.20, Graph[:, 7], color='#ddd618', edgecolor='k', width=0.10, hatch="+", label="GRU")
        ax.bar(X + 0.30, Graph[:, 8], color='#6ba353', edgecolor='k', width=0.10, hatch="+", label="LSTM")
        ax.bar(X + 0.40, Graph[:, 9], color='b', edgecolor='k', width=0.10, hatch="*", label="IOOA-SRF-CAALSTM")
        plt.xticks(X + 0.10, (
            "Linear","Tanh","Softmax","Relu","Sigmoid"), rotation=7)
        plt.ylabel(Terms[Graph_Term[j]])
        plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15), ncol=3, fancybox=True, shadow=True)
        path1 = "./Results/Learn_%s_bar_Act.png" % (Terms[Graph_Term[j]])
        plt.xlabel('Activation Function')
        plt.savefig(path1)
        plt.show()

def Plot_Batch():
    Eval = np.load('Eval_Batch.npy', allow_pickle=True)
    Terms = np.asarray(['MAE', 'MD', 'MASE', 'SMAPE', 'RMSE', 'ONE-NORM', 'TWO-NORM'])
    Algorithm = ['TERMS', 'POA-SRF-CAALSTM', 'HCO-SRF-CAALSTM', 'DO-SRF-CAALSTM', 'OOA-SRF-CAALSTM', 'IOOA-SRF-CAALSTM']
    Classifier = ['TERMS',  'DNN', 'DBN', 'GRU','LSTM', 'IOOA-SRF-CAALSTM']
    Activation = ["4","8","16","32","48"]
    for a in range(2):
        Graph_T = [2,4]
        Graph_Term = np.array([Graph_T[a]]).astype(int)
        value = Eval[:, :,Graph_Term[0]]
        # value[:, :-1] = value[:, :-1]
        Table = PrettyTable()
        Table.add_column(Algorithm[0], Activation[0:])
        for j in range(len(Algorithm) - 1):
            Table.add_column(Algorithm[j + 1], value[:, j])
        print('---------------------------------------------------Batch Size',  '- Algorithm Comparison -',
              Terms[Graph_Term],
              '--------------------------------------------------')
        print(Table)

        Table = PrettyTable()
        Table.add_column(Classifier[0], Activation[0:])
        for j in range(len(Classifier) - 1):
            Table.add_column(Classifier[j + 1], value[:, j+5])
        print('---------------------------------------------------Batch Size',  ' - Classifier Comparison -',
              Terms[Graph_Term],
              '--------------------------------------------------')
        print(Table)


def Plot_Fitness():
    conv = np.load('Fitness.npy', allow_pickle=True)
    Statistics = ['BEST', 'WORST', 'MEAN', 'MEDIAN', 'STD']
    Algorithm = [ 'POA-SRF-CAALSTM', 'HCO-SRF-CAALSTM', 'DO-SRF-CAALSTM', 'OOA-SRF-CAALSTM', 'IOOA-SRF-CAALSTM']

    Value = np.zeros((conv.shape[0], 5))
    for j in range(conv.shape[0]):
        Value[j, 0] = np.min(conv[j, :])
        Value[j, 1] = np.max(conv[j, :])
        Value[j, 2] = np.mean(conv[j, :])
        Value[j, 3] = np.median(conv[j, :])
        Value[j, 4] = np.std(conv[j, :])

    Table = PrettyTable()
    Table.add_column("ALGORITHMS", Statistics)
    for j in range(len(Algorithm)):
        Table.add_column(Algorithm[j], Value[j, :])
    print('--------------------------------------------------Statistical Analysis--------------------------------------------------')
    print(Table)

    iteration = np.arange(conv.shape[1])
    plt.plot(iteration, conv[0, :], color='r', linewidth=3, marker='>', markerfacecolor='blue', markersize=8,
             label="POA-SRF-CAALSTM")
    plt.plot(iteration, conv[1, :], color='g', linewidth=3, marker='>', markerfacecolor='red', markersize=8,
             label="HCO-SRF-CAALSTM")
    plt.plot(iteration, conv[2, :], color='b', linewidth=3, marker='>', markerfacecolor='green', markersize=8,
             label="DO-SRF-CAALSTM")
    plt.plot(iteration, conv[3, :], color='m', linewidth=3, marker='>', markerfacecolor='yellow', markersize=8,
             label="OOA-SRF-CAALSTM")
    plt.plot(iteration, conv[4, :], color='k', linewidth=3, marker='>', markerfacecolor='cyan', markersize=8,
             label="IOOA-SRF-CAALSTM")
    plt.xlabel('Iteration')
    plt.ylabel('Cost Function')
    plt.legend(loc=1)
    path1 = "./Results/conv.png"
    plt.savefig(path1)
    plt.show()


def statistical_analysis(v):
    a = np.zeros((5))
    a[0] = np.min(v)
    a[1] = np.max(v)
    a[2] = np.mean(v)
    a[3] = np.median(v)
    a[4] = np.std(v)
    return a

if __name__ == "__main__":
    Plot_Activation_function()
    Plot_Batch()
    Plot_Fitness()