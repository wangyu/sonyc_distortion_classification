import numpy as np
import matplotlib.pyplot as plt

#KNN:  dataSet is all the datapoints(including labeled and unlabeled), query is One datapoint. we need to return a list of datapoints that are the top k most closest datapoints.
def navie_knn(dataSet, query, k):
    numSamples = dataSet.shape[0]

    ## calculate Euclidean distance between the query entry and the rest datapoints.
    diff = np.tile(query, (numSamples, 1)) - dataSet
    squaredDiff = diff ** 2
    squaredDist = np.sum(squaredDiff, axis=1)  # sum is performed by row

    ## Sort the distance
    sortedDistIndices = np.argsort(squaredDist)
    if k > len(sortedDistIndices): # make sure we have enough data N greater than the K in this naive KNN
        k = len(sortedDistIndices)
        print("***Warning: Not Enough Data points for KNN!***")

    return sortedDistIndices[0:k]

# build a big graph (normalized weight matrix)
def buildGraph(MatX, kernel_type, rbf_sigma=None, knn_num_neighbors=None):
    num_samples = MatX.shape[0]
    affinity_matrix = np.zeros((num_samples, num_samples), np.float32)
    if kernel_type == 'rbf':
        if rbf_sigma == None:
            raise ValueError('You should input a sigma of rbf kernel!')
        for i in range(num_samples):
            row_sum = 0.0
            for j in range(num_samples):
                diff = MatX[i, :] - MatX[j, :]
                affinity_matrix[i][j] = np.exp(sum(diff ** 2) / (-2.0 * rbf_sigma ** 2))
                row_sum += affinity_matrix[i][j]
            affinity_matrix[i][:] /= row_sum
    elif kernel_type == 'knn':
        if knn_num_neighbors == None:
            raise ValueError('You should input a k of knn kernel!')
        for i in range(num_samples):
            k_neighbors = navie_knn(MatX, MatX[i, :], knn_num_neighbors)
            affinity_matrix[i][k_neighbors] = 1.0 / knn_num_neighbors
    else:
        raise NameError('Not support kernel type! You can use knn or rbf!')

    return affinity_matrix # square matrix, dim = N * N


# label propagation
def labelPropagation(Mat_Label, Mat_Unlabel, labels, kernel_type='rbf', rbf_sigma=1.5, \
                     knn_num_neighbors=10, max_iter=500, tol=1e-3):
    # initialize
    num_label_samples = Mat_Label.shape[0]
    num_unlabel_samples = Mat_Unlabel.shape[0]
    num_samples = num_label_samples + num_unlabel_samples
    labels_list = np.unique(labels)
    num_classes = len(labels_list)

    # reserve 10% of labeled data for testing
    holdup_ratio = 0.1
    num_holdup_samples = num_label_samples * holdup_ratio
    holdup_idx = np.random.choice(num_label_samples, int(num_holdup_samples), replace=False)
    train_idx  = np.delete(np.arange(num_label_samples), holdup_idx, axis=0)

    num_label_samples -= int(num_holdup_samples)

    #
    Mat_holdup = Mat_Label[holdup_idx]
    TrueLabel_holdup = labels[holdup_idx]

    Mat_Label  = Mat_Label[train_idx]
    TrueLabel_Mat_Label = labels[train_idx]

    MatX = np.vstack((Mat_Label, Mat_Unlabel, Mat_holdup))

    clamp_data_label = np.zeros((int(num_label_samples), num_classes), np.float32)
    for i in range(num_label_samples):
        clamp_data_label[i][int(TrueLabel_Mat_Label[i])] = 1.0

    # initialize the label function that contains all the 0s and 1s
    label_function = np.zeros((num_samples, num_classes), np.float32)
    # assign the pre-labeled datapoints to the label_function (whole set)
    label_function[0: num_label_samples] = clamp_data_label
    # the rest is just gonna be -1
    label_function[num_label_samples: num_samples] = -1


    # graph construction
    affinity_matrix = buildGraph(MatX, kernel_type, rbf_sigma, knn_num_neighbors)

    convergence_arr = []

    # start to propagation
    iter = 0;
    pre_label_function = np.zeros((num_samples, num_classes), np.float32) # dummy place holder for label_function
    changed = np.abs(pre_label_function - label_function).sum() # each iteration we try to minimize the label_function
    while iter < max_iter and changed > tol:
        if iter % 1 == 0:
            print("---> Iteration %d/%d, changed: %f" % (iter, max_iter, changed))

        if iter != 0:
            convergence_arr.append(changed)

        pre_label_function = label_function
        iter += 1

        # propagation
        label_function = np.dot(affinity_matrix, label_function)

        # clamp back on the true labels that were manually assigned.
        # TODO:
        # we want to calculate entropy at this point and give manual input!!! (For a faster convergence!)
        #
        label_function[0: num_label_samples] = clamp_data_label

        # check converge
        changed = np.abs(pre_label_function - label_function).sum()

    # get terminate label of unlabeled data
    unlabel_data_labels = np.zeros(num_unlabel_samples)
    for i in range(num_unlabel_samples):
        unlabel_data_labels[i] = np.argmax(label_function[i + num_label_samples])

    # Plot convergence change graph
    # plt.plot(convergence_arr)
    # plt.xlabel("Iteration")
    # plt.ylabel("Label Convergence")
    # plt.title("Change of Label_Functions throughout Iterations")
    # plt.show()

    # calculate accuracy, with TrueLabel_holdup
    predict_holdup = unlabel_data_labels[-int(num_holdup_samples):]

    correct = 0
    for idx, l in np.ndenumerate(predict_holdup):
        if TrueLabel_holdup[idx] == l:
            correct += 1

    print('Number of Corrected Prediction = ' + str(correct))
    print('Number of Mistake Prediction = ' + str(num_holdup_samples - correct))
    error_rate = (num_holdup_samples - correct)/num_holdup_samples
    print('Accuracy = '+str(error_rate))

    return unlabel_data_labels, error_rate