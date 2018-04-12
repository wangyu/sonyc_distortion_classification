import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.semi_supervised import label_propagation
from sklearn.metrics import classification_report, confusion_matrix
import pickle


# read in prepared data
positive_xy_file = open('../positive_xy.pickle', 'rb')
positive_xy = pickle.load(positive_xy_file)

negative_xy_file = open('../negative_xy.pickle', 'rb')
negative_xy = pickle.load(negative_xy_file)

# stack both positive and negative data:
train_xy = np.vstack((positive_xy, negative_xy))

# seperate into X and y
X = train_xy[:,:128]    # first 128 digits
y = train_xy[:,128:129] # last digit, 0 or 1

# reserve some seeds for initial labeled training data
# we assume that these data are correctly labeled!
init_seeds = 20 # for both positive and negative labels

n_total_samples = train_xy.shape[0]
n_labeled_points = 2 * init_seeds
max_iterations = 50 # test for 10 runs

# keep a list of indices for unlabeled items:
# indices for the init_seeds of positive and negative should be REMOVED!!!
# positive: 0 ~ 20
# negative: 171 ~ 171+20 = 191
indices = np.arange(n_total_samples)
unlabeled_indices = np.delete(indices, np.array([np.arange(0,21), np.arange(171,192)]), 0) # now indices inside this list are unlabled

# start iterations:
for i in range(max_iterations):

    if len(unlabeled_indices) == 0:
        print("No unlabeled items left to label.")
        break

    y_train = np.copy(y)
    y_train[unlabeled_indices] = -1 # does not matter

    lp_model = label_propagation.LabelSpreading(gamma=0.25, max_iter=5)
    lp_model.fit(X, y_train)

    # the following is for performance inspection:
    predicted_labels = lp_model.transduction_[unlabeled_indices]
    true_labels = y[unlabeled_indices]

    print("Iteration %i %s" % (i, 70 * "_"))
    print("Label Spreading model: %d labeled & %d unlabeled (%d total)" % (n_labeled_points, n_total_samples - n_labeled_points, n_total_samples))
    print(classification_report(true_labels, predicted_labels))

    cm = confusion_matrix(true_labels, predicted_labels, labels=lp_model.classes_)
    print("Confusion matrix")
    print(cm)


    # compute the entropies of transduced label distributions
    pred_entropies = stats.distributions.entropy(lp_model.label_distributions_.T)

    # select up to 5 digit examples that the classifier is most uncertain about
    uncertainty_index = np.argsort(pred_entropies)[::-1]
    uncertainty_index = uncertainty_index[np.in1d(uncertainty_index, unlabeled_indices)][:5]

    for index, classified_index in enumerate(uncertainty_index):
        # labeling these 5 points and remote from labeled set

        # TODO!
        # add the wait-and-input loop here!!!
        # we have to manually type in the label here!!! Change this line!!!
        # true_new_label =
        print("Please label: index="+str(classified_index))
        # provide sensor id and timestamp
        true_new_label = y[classified_index]


        # assign the true_new_label to the entry:
        y[classified_index] = true_new_label

        # remove this classified_index from the unlabeled_indices array:
        index_of_the_entry_to_be_removed = np.where(unlabeled_indices == classified_index)
        unlabeled_indices = np.delete(unlabeled_indices, index_of_the_entry_to_be_removed)

        # we got one more data-point labeled!!!

        n_labeled_points += 1

    print(str(len(unlabeled_indices))+" more data-points to be classified...")