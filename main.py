import numpy as np
import matplotlib.pyplot as plt
from draw_fc import plot_data,plot_confusion_matrix
from sklearn.metrics import  accuracy_score
from tqdm import tqdm

from MahalanobisClassifier_fc import MahalanobisClassifier
from data_loader_fc import test_EXMNIST,test_CIFAR10
from dimension_reduce_fc import *
import csv

def main():
    train_data_orign,train_labels_orign,test_data_orign,test_labels_orign=test_CIFAR10()
    score=[]
    begai_dimession=10
    nn_components=begai_dimession
    # test_times=30
    # interval=5
    # for i in tqdm(range(test_times),desc='tuiduan', leave=True,dynamic_ncols=True):
    #     train_data,train_labels,test_data,test_labels=train_data_orign,train_labels_orign,test_data_orign,test_labels_orign
    #     pca=MDS()
    #     pca = pca(n_components=nn_components)
    #     train_data =  pca.fit_transform(train_data)
    #     test_data = pca.transform(test_data)
        
    # #  MahalanobisClassifier Classifier
    #     clf = MahalanobisClassifier(train_data,train_labels)
    #     # Evaluate models on given test dataset
    #     pred_probs,pred_class = clf.predict_probability(test_data,np.unique(test_labels))
    #     accuracy=accuracy_score(np.asarray(test_labels), pred_class)
    #     score.append(accuracy)
    #     nn_components=nn_components+interval

    # score=[100 * i for i in list(score)]

    # plt.plot(np.arange(begai_dimession, begai_dimession+test_times*interval,interval),score,linewidth=2.5, marker='*', markersize=8)

    # with open('C:\\Users\songyu\\Desktop\\Cifar_10-main\\my_array.csv', 'w', newline='') as file:
    #     writer = csv.writer(file)
    #     writer.writerow(score)


    train_data,train_labels,test_data,test_labels=train_data_orign,train_labels_orign,test_data_orign,test_labels_orign
    pca=PCA()
    pca = pca(n_components=100)
    train_data =  pca.fit_transform(train_data)
    test_data = pca.transform(test_data)
    
#  MahalanobisClassifier Classifier
    clf = MahalanobisClassifier(train_data,train_labels)
    # Evaluate models on given test dataset
    pred_probs,pred_class = clf.predict_probability(test_data,np.unique(test_labels))
    accuracy=accuracy_score(np.asarray(test_labels), pred_class)
    score.append(accuracy)

    class_names=['0','1','2','3','4','5','6','7','8','9']

    plt.figure()
    plot_confusion_matrix(np.asarray(test_labels), pred_class, classes=class_names,
                              normalize=True)
    title='sd'
    print(f'{title}: {accuracy_score(np.asarray(test_labels), pred_class)}')
    plot_data(
        np.asarray(train_data)[:, :2], train_labels,
        sample_icon='*',
        sample_color=['INDIANRED', 'DEEPPINK', 'MEDIUMVIOLETRED', 'DARKORANGE', 'GOLDENROD',
                      'MAGENTA', 'DARKSLATEBLUE', 'GREENYELLOW', 'STEELBLUE', 'DARKSLATEGRAY'],
        title='Visualization of Sparse Matrices After Dimensionality Reduction with PCA on CIFAR-10')
    plt.show()

main()
