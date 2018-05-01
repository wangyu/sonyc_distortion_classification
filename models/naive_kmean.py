from __future__ import print_function

import numpy as np
import tensorflow as tf
from tensorflow.contrib.factorization import KMeans
import pickle
import matplotlib.pyplot as plt

# Remove use of GPU
# import os
# os.environ["CUDA_VISIBLE_DEVICES"] = ""

# import SoNYC data:
sonyc_length = 128
class_length = 1
print("** Loading SoNYC datasets **")
negative_xy = pickle.load(open("../data/negative_xy.pickle", "rb"))
positive_xy = pickle.load(open("../data/positive_xy.pickle", "rb"))

# Choose the data source
# x_pool_file = "../data/X_pool.pickle" # 1000 * 128
# x_pool_file = "../data/X_pool_10000_new.pickle" # 10000 * 132
# x_pool_file = "../data/X_pool_100000_new.pickle"
# x_pool_file = "../data/X_pool_100000_random.pickle"
X_pool = pickle.load(open(x_pool_file, "rb"))

# Modify the SoNYC data (a little bit) just for this model:
X_pool = X_pool[:,:sonyc_length] # number_of_unlabeled * 128
labeled_data = np.vstack((positive_xy, negative_xy)) # 600 * 129
labels = labeled_data[:, sonyc_length] # 600 * 1
Mat_Label = np.split(labeled_data, [sonyc_length], axis=1)[0] # 600 * 128

full_data_x = np.vstack((Mat_Label, X_pool))

# Parameters
kmean_iterations = 200 # Total steps to train
k = 2 # The number of clusters
num_classes = 2 # good or bad audio
num_features = 128 # Length of each audio embedding = 128

X = tf.placeholder(tf.float32, shape=[None, num_features])

kmeans = KMeans(inputs=X,
                num_clusters=k,
                distance_metric='cosine',
                use_mini_batch=True)

# use tensorflow 1.7
(all_scores, cluster_idx, scores, cluster_centers_initialized, init_op, train_op) = kmeans.training_graph()

cluster_idx = cluster_idx[0]
avg_distance = tf.reduce_mean(scores)

init_vars = tf.global_variables_initializer()

sess = tf.Session()

sess.run(init_vars, feed_dict={X: full_data_x})
sess.run(init_op, feed_dict={X: full_data_x})

error_rates = []
display_steps = 10
for i in range(1, kmean_iterations + 1):
    _, d, idx = sess.run([train_op, avg_distance, cluster_idx], feed_dict={X: full_data_x})

    part1 = idx[0:300]
    part2 = idx[301:600]
    sumPart1 = np.sum(part1)
    sumPart2 = np.sum(part2)
    # Label the cluster
    if sumPart1 >= sumPart2: # as it is
        correct_part1 = sumPart1
        correct_part2 = 300-sumPart2
    else: # reverse
        correct_part1 = 300-sumPart1
        correct_part2 = sumPart2

    # Calculate error rate:
    error_rate = 1 - (correct_part1 + correct_part2)/600
    error_rates.append(error_rate)

    if i % display_steps == 0 or i == 1:
        print("Step %i, Avg Distance: %f" % (i, d))
        print('Error rate: %f' % error_rate)

# Plot error rate
plt.plot(error_rates)
plt.xlabel("Iteration")
plt.ylabel("Error Rate")
plt.title("Error Rate in K-Means Iterations. \nUnlabel File:"+x_pool_file)
plt.axis([-10, kmean_iterations+10, 0, 1])
plt.show()

