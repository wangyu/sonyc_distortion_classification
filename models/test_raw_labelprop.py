import pickle
import numpy as np
from labelprop_raw_algorithm import labelPropagation
import matplotlib.pyplot as plt
from functools import reduce

def loadSonycData():
    print("** Loading SoNYC datasets **")
    negative_xy = pickle.load(open("../data/negative_xy.pickle", "rb"))
    positive_xy = pickle.load(open("../data/positive_xy.pickle", "rb"))
    X_pool = pickle.load(open("../data/X_pool.pickle", "rb"))
    # id_pool = pickle.load(open("../data/id_pool.pickle", "rb"))

    # Select Data
    # negative_xy = negative_xy[:,:128]
    # positive_xy = positive_xy[:,:128]
    # negative_xy = negative_xy.astype(np.int)
    # positive_xy = positive_xy.astype(np.int)

    labeled_with_label = np.vstack((positive_xy, negative_xy))
    labels = labeled_with_label[:, 128]
    Mat_Label = np.split(labeled_with_label, [128], axis=1)[0]
    Mat_Unlabel = X_pool[:,:128]

    return Mat_Label, labels, Mat_Unlabel

if __name__ == "__main__":
    Mat_Label, labels, Mat_Unlabel = loadSonycData()

    error_rates = []

    number_of_iterations = 10
    for i in range(number_of_iterations):
        unlabel_data_labels, error_rate = labelPropagation(Mat_Label, Mat_Unlabel, labels,
                                                           kernel_type='knn',
                                                           knn_num_neighbors=50,
                                                           max_iter=100)
        error_rates.append(error_rate)

    plt.plot(error_rates)
    plt.xlabel("Run")
    plt.ylabel("Error Rate")
    plt.title("Error Rate in Runs")
    plt.show()

    print("Average Error Rate: ")
    print(reduce(lambda x, y: x + y, error_rates) / len(error_rates))