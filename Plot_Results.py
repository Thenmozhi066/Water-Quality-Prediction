from itertools import cycle
from sklearn.metrics import roc_curve
import numpy as np
from prettytable import PrettyTable
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon

no_of_dataset = 2

def statistical_analysis(v):
    a = np.zeros((5))
    a[0] = np.min(v)
    a[1] = np.max(v)
    a[2] = np.mean(v)
    a[3] = np.median(v)
    a[4] = np.std(v)
    return a


def addlabels(x, y):
    for i in range(len(x)):
        plt.text(i - 0.20, np.round(y[i], 3) / 2, str(np.round(y[i], 2)) + '%')

def plot_results_conv():
    for a in range(no_of_dataset):
        conv = np.load('Fitness.npy', allow_pickle=True)[a]
        Statistics = ['BEST', 'WORST', 'MEAN', 'MEDIAN', 'STD']
        Algorithm = ['POA-SRF-CAALSTM', 'HCO-SRF-CAALSTM', 'DO-SRF-CAALSTM', 'OOA-SRF-CAALSTM', 'IOOA-SRF-CAALSTM']

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
        print('--------------------------------------------------Dataset_' + str(
            a + 1) + ' - Statistical Analysis--------------------------------------------------')
        print(Table)

        iteration = np.arange(conv.shape[1])
        plt.plot(iteration, conv[0, :], color='m', linewidth=3, marker='.', markerfacecolor='red', markersize=12,
                 label='POA-SRF-CAALSTM')
        plt.plot(iteration, conv[1, :], color='c', linewidth=3, marker='.', markerfacecolor='green', markersize=12,
                 label='HCO-SRF-CAALSTM')
        plt.plot(iteration, conv[2, :], color='b', linewidth=3, marker='.', markerfacecolor='cyan', markersize=12,
                 label='DO-SRF-CAALSTM')
        plt.plot(iteration, conv[3, :], color='r', linewidth=3, marker='.', markerfacecolor='magenta', markersize=12,
                 label='OOA-SRF-CAALSTM')
        plt.plot(iteration, conv[4, :], color='k', linewidth=3, marker='*', markerfacecolor='black', markersize=12,
                 label='IOOA-SRF-CAALSTM')
        plt.xlabel('Iteration')
        plt.ylabel('Cost Function')
        plt.legend(loc=1)
        path = "./Results/Conv_%s.png" % (str(a + 1))
        plt.savefig(path)
        plt.show()


def Plot_ROC():
    lw = 2
    cls = ['DNN', 'DBN', 'GRU', 'LSTM', 'IOOA-SRF-CAALSTM']
    colors1 = cycle(["#65fe08", "#4e0550", "#f70ffa", "#a8a495", "#004577", ])
    for n in range(no_of_dataset):
        for i, color in zip(range(5), colors1):  # For all classifiers
            Predicted = np.load('roc_score.npy', allow_pickle=True)[n][i].astype('float')
            Actual = np.load('roc_act.npy', allow_pickle=True)[n][i].astype('int')
            false_positive_rate1, true_positive_rate1, threshold1 = roc_curve(Actual[:, -1], Predicted[:, -1].ravel())
            plt.plot(
                false_positive_rate1,
                true_positive_rate1,
                color=color,
                lw=lw,
                label="{0}".format(cls[i]),
            )
        plt.plot([0, 1], [0, 1], "k--", lw=lw)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("ROC Curve")
        plt.legend(loc="lower right")
        path = "./Results/roc_%s.png" % (str(n + 1))
        plt.savefig(path)
        plt.show()

def Plot_table():
    for i in range(no_of_dataset):
        Eval = np.load('Eval_ALL_Batch.npy', allow_pickle=True)[i]
        Graph_Term = np.array([0]).astype(int)
        Graph_Term_Array = np.array(Graph_Term)
        Algorithm = ['TERMS', 'POA-SRF-CAALSTM', 'HCO-SRF-CAALSTM', 'DO-SRF-CAALSTM', 'OOA-SRF-CAALSTM', 'IOOA-SRF-CAALSTM']
        Classifier = ['TERMS', 'DNN', 'DBN', 'GRU', 'LSTM', 'IOOA-SRF-CAALSTM']
        variation  = ['100', '200', '300', '400', '500']
        value = Eval[:, :, 4:]
        Table = PrettyTable()
        Table.add_column(Algorithm[0], variation[0:])
        for j in range(len(Algorithm) - 1):
            Table.add_column(Algorithm[j + 1], value[:, j, Graph_Term])
        print('--------------------------------------------------  Batch Size - Algorithm Comparison -',
              'Accuracy- Dataset-', str(i+1),
              '--------------------------------------------------')
        print(Table)
        Table = PrettyTable()
        Table.add_column(Classifier[0], variation[0:])
        for j in range(len(Classifier) - 1):
            Table.add_column(Classifier[j + 1], value[:, len(Algorithm) + j - 1, Graph_Term])
        print('--------------------------------------------------- Batch Size - Classifier Comparison -',
              'Accuracy- Dataset-', str(i+1),
              '--------------------------------------------------')
        print(Table)



def Plot_Met():
    eval = np.load('Eval_ALL_Act.npy', allow_pickle=True)
    Terms = ['Accuracy', 'Sensitivity', 'Specificity', 'Precision', 'FPR', 'FNR', 'NPV', 'FDR', 'F1_score', 'MCC',
             'FOR', 'pt',
             'ba', 'fm', 'bm', 'mk', 'PLHR', 'lrminus', 'dor', 'prevalence', 'TS']
    Graph_Term = [0, 1, 2, 3, 4, 5]
    Table_Terms = [0, 13, 14, 15, 16]
    Activation = ['1', '2', '3', '4', '5']
    Dataset = ['Dataset1', 'Dataset2', 'Dataset3', 'Dataset4', 'Dataset5']

    for n in range(eval.shape[0]):
        for j in range(len(Graph_Term)):
            Graph = np.zeros((eval.shape[1], eval.shape[2]))
            for k in range(eval.shape[1]):
                for l in range(eval.shape[2]):
                    Graph[k, l] = eval[n, k, l, Graph_Term[j] + 4]

            Batch_size = ['Linear', 'ReLU', 'Tanh', 'Softmax', 'Sigmoid']
            colors = ['#00ffff', '#fe01b1', '#0cff0c', '#be03fd', 'k']
            mtd_Graph = Graph[:, 5:]

            Methods = [ 'DNN', 'DBN', 'GRU', 'LSTM', 'IOOA-SRF-CAALSTM']
            fig = plt.figure()
            ax = fig.add_axes([0.15, 0.1, 0.7, 0.8])
            triangle_width = 0.15
            for y, algorithm in enumerate(Methods):
                for x, batch_size in enumerate(Batch_size):
                    x_pos = x + y * triangle_width
                    value = mtd_Graph[x, y]
                    color = colors[y]
                    triangle_points = np.array(
                        [[x_pos - triangle_width / 2, 0], [x_pos + triangle_width / 2, 0], [x_pos, value]])
                    triangle = Polygon(triangle_points, ec='k', closed=True, color=color, linewidth=1)
                    ax.add_patch(triangle)
                    ax.plot(x_pos, value, marker='.', color=color, markersize=10)  # 'orange' , marker=(6, 2, 0)
                ax.plot(x_pos, value, marker='.', color=color, markersize=10, label=Methods[y])

            ax.set_xticks(np.arange(len(Batch_size)) + (len(Methods) * triangle_width) / 2)
            ax.set_xticklabels([str(size) for size in Batch_size])
            plt.legend(loc=1)

            plt.xlabel('Activation Function', fontname="Arial", fontsize=12, fontweight='bold', color='#35530a')
            plt.ylabel(Terms[Graph_Term[j]], fontname="Arial", fontsize=12, fontweight='bold', color='#35530a')
            path = "./Results/%s_Activation_Function_%s_bar.png" % (Dataset[n], Terms[Graph_Term[j]])
            plt.savefig(path)
            plt.show()
def Plot_table():
    for i in range(no_of_dataset):
        Eval = np.load('Eval_ALL_Batch.npy', allow_pickle=True)[i]
        Terms = ['Accuracy', 'Precision', 'FOR', 'PT', 'BM', 'MK', 'Prevalence', 'TS']
        # Terms = np.asarray(
        #     ['Accuracy', 'Sensitivity', 'Specificity', 'Precision', 'FPR', 'FNR', 'NPV', 'FDR', 'F1_score',
        #      'MCC', 'FOR', 'PT', 'BA', 'FM', 'BM', 'MK', 'PLHR', 'Lrminus', 'DOR', 'Prevalence', 'TS'])
        Graph_Term = np.array([0]).astype(int)
        Graph_Term_Array = np.array(Graph_Term)
        Algorithm = ['TERMS', 'POA-SRF-CAALSTM', 'HCO-SRF-CAALSTM', 'DO-SRF-CAALSTM', 'OOA-SRF-CAALSTM', 'IOOA-SRF-CAALSTM']
        Classifier = ['TERMS',  'DNN', 'DBN', 'GRU', 'LSTM', 'IOOA-SRF-CAALSTM']
        variation  = ['100', '200', '300', '400', '500']
        value = Eval[:, :, 4:]
        Table = PrettyTable()
        Table.add_column(Algorithm[0], variation[0:])
        for j in range(len(Algorithm) - 1):
            Table.add_column(Algorithm[j + 1], value[:, j, Graph_Term])
        print('--------------------------------------------------  Batch Size - Algorithm Comparison -',
              'Accuracy- Dataset-', str(i+1),
              '--------------------------------------------------')
        print(Table)
        Table = PrettyTable()
        Table.add_column(Classifier[0], variation[0:])
        for j in range(len(Classifier) - 1):
            Table.add_column(Classifier[j + 1], value[:, len(Algorithm) + j - 1, Graph_Term])
        print('--------------------------------------------------- Batch Size - Classifier Comparison -',
              'Accuracy- Dataset-', str(i+1),
              '--------------------------------------------------')
        print(Table)


def Plot_Alg():
    for i in range(no_of_dataset):
        Eval = np.load('Eval_ALL_Act.npy', allow_pickle=True)[i]
        Terms = ['Accuracy', 'Sensitivity', 'Specificity', 'Precision', 'FPR', 'FNR', 'NPV', 'FDR', 'F1_score', 'MCC',
                 'FOR', 'pt',
                 'ba', 'fm', 'bm', 'mk', 'PLHR', 'lrminus', 'dor', 'prevalence', 'TS']
        Graph_Term = [0, 1, 2, 3, 4, 5]
        # Algorithm = ['TERMS','TFMOA-ARAN', 'SCO-ARAN', 'FFO-ARAN', 'GSOA-ARAN', 'IGSOA-ARAN']
        # Classifier = ['TERMS','CNN', 'Densenet', 'MobileNet', 'RAN', 'IGSOA-ARAN']
        Eval = np.load('Eval_ALL_Act.npy', allow_pickle=True)[i]
        BATCH = [ 1, 2, 3, 4, 5]
        for j in range(len(Graph_Term)):
            Graph = np.zeros((Eval.shape[0], Eval.shape[1]))
            for k in range(Eval.shape[0]):
                for l in range(Eval.shape[1]):
                    Graph[k, l] = Eval[k, l, Graph_Term[j] + 4]
            X = np.arange(5)
            plt.plot(BATCH, Graph[:, 0], '-', color='purple', linewidth=3, marker='.', markerfacecolor='k',
                     markersize=16,
                     label='POA-SRF-CAALSTM')
            plt.plot(BATCH, Graph[:, 1], '-', color='DeepPink', linewidth=3, marker='.', markerfacecolor='k',
                     markersize=16,
                     label='HCO-SRF-CAALSTM')
            plt.plot(BATCH, Graph[:, 2], '-', color='cyan', linewidth=3, marker='.', markerfacecolor='k',
                     markersize=16,
                     label='DO-SRF-CAALSTM')
            plt.plot(BATCH, Graph[:, 3], '-', color='red', linewidth=3, marker='.', markerfacecolor='k', markersize=16,
                     label='OOA-SRF-CAALSTM')
            plt.plot(BATCH, Graph[:, 4], '-', color='k', linewidth=3, marker='.', markerfacecolor='white',
                     markersize=16,
                     label="IOOA-SRF-CAALSTM")
            plt.xlabel('Activation Function')
            plt.xticks(X + 1, ('Linear', 'ReLU', 'Tanh', 'Softmax', 'Sigmoid'))
            # plt.xticks(BATCH, ('4', '8', '16', '32', '48'))
            plt.ylabel(Terms[Graph_Term[j]])
            plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15),
                       ncol=3, fancybox=True, shadow=True)
            path = "./Results/Dataset_%s_%s_line_1.png" % ((str(i + 1)), Terms[Graph_Term[j]])
            plt.savefig(path)
            plt.show()


if __name__ == '__main__':
    plot_results_conv()
    Plot_ROC()
    Plot_Met()
    Plot_Alg()
    Plot_table()