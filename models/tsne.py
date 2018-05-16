import pickle
import numpy as np
from numpy import linalg
from numpy.linalg import norm
from scipy.spatial.distance import squareform, pdist
import sklearn
from sklearn.manifold import TSNE
from sklearn.datasets import load_digits
from sklearn.preprocessing import scale
import matplotlib.pyplot as plt
import matplotlib.patheffects as PathEffects
import matplotlib

# We'll hack a bit with the t-SNE code in sklearn 0.15.2.
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.manifold.t_sne import (_joint_probabilities, _kl_divergence)

# We import seaborn to make nice plots.
# import seaborn as sns
# sns.set_style('darkgrid')
# sns.set_palette('muted')
# sns.set_context("notebook", font_scale=1.5, rc={"lines.linewidth": 2.5})

# import SoNYC data:
sonyc_length = 128
class_length = 1
print("** Loading SoNYC datasets **")
negative_xy = pickle.load(open("../data/negative_xy.pickle", "rb"))
positive_xy = pickle.load(open("../data/positive_xy.pickle", "rb"))

# Choose the data source
# x_pool_file = "X_pool" # 1000 * 128
# x_pool_file = "X_pool_10000_new" # 10000 * 132
# x_pool_file = "X_pool_100000_new"
x_pool_file = "X_pool_100000_random" # Warning: This pool is HUGE!
X_pool = pickle.load(open('../data/'+x_pool_file+'.pickle', "rb"))

# randomly sample 10000 from this HUGE dataset
number_of_random_sample = 20000
if x_pool_file == "X_pool_100000_random":
    randomIndex = np.random.choice(X_pool.shape[0], number_of_random_sample)
    X_pool = X_pool[randomIndex]

# Modify the SoNYC data (a little bit) just for this model:
X_pool = X_pool[:,:sonyc_length] # number_of_unlabeled * 128
labeled_data = np.vstack((positive_xy, negative_xy)) # 600 * 129
labels = labeled_data[:, sonyc_length] # 600 * 1
Mat_Label = np.split(labeled_data, [sonyc_length], axis=1)[0] # 600 * 128

full_data_x = np.vstack((Mat_Label, X_pool))

# Random state.
RS = 20180101
use_random_state = False
# Parameters
try_perplexity = 50
number_iterations = 1000

if use_random_state:
    tsne_result = TSNE(random_state=RS).fit_transform(full_data_x)
else:
    tsne_result = TSNE(n_components=2,
                       verbose=1,
                       perplexity=try_perplexity,
                       n_iter=number_iterations).fit_transform(full_data_x)

# Visualize the Result:
x_pos = tsne_result[0:300,0]
y_pos = tsne_result[0:300,1]

x_neg = tsne_result[301:600,0]
y_neg = tsne_result[301:600,1]

x_unlabel = tsne_result[601:X_pool.shape[0],0]
y_unlabel = tsne_result[601:X_pool.shape[0],1]

# unlabel data points are too many, sample only a handful of them for visualization:
number_of_unlabeled_to_show = int(x_unlabel.shape[0] / 10)
unlabel_to_show_index = np.random.choice(x_unlabel.shape[0], number_of_unlabeled_to_show)

x_unlabel = x_unlabel[unlabel_to_show_index]
y_unlabel = y_unlabel[unlabel_to_show_index]

fig = plt.figure()
plt.scatter(x_pos,y_pos,c='g',s=30,label='pos')
plt.scatter(x_neg,y_neg,c='r',s=30,label='neg')
plt.scatter(x_unlabel,y_unlabel,c='b',s=1,label='unlabel')
plt.legend(loc='upper left')

if x_pool_file == "X_pool_100000_random":
    plt.title("Preplexity:" + str(try_perplexity)
              + "\nIterations:" + str(number_iterations)
              + "\nUnlabel File:" + str(x_pool_file)
              + "\nRandom Sample:" + str(number_of_random_sample))
else:
    plt.title("Preplexity:" + str(try_perplexity)
              + "\nIterations:" + str(number_iterations)
              + "\nUnlabel File:" + str(x_pool_file))

plt.savefig('result_plots/tsne_'+x_pool_file+'_p'+str(try_perplexity)+'.png', dpi=120)
plt.show()


# Save the t-SNE low dimensional data for labelpropagation:
x_unlabel = tsne_result[601:X_pool.shape[0],0]
y_unlabel = tsne_result[601:X_pool.shape[0],1]

pickle.dump(x_unlabel, open("x_unlabel.pickle", 'wb'))
pickle.dump(y_unlabel, open("y_unlabel.pickle", 'wb'))
pickle.dump(x_neg, open("x_neg.pickle", 'wb'))
pickle.dump(y_neg, open("y_neg.pickle", 'wb'))
pickle.dump(x_pos, open("x_pos.pickle", 'wb'))
pickle.dump(y_pos, open("y_pos.pickle", 'wb'))

# Below, show the connection
'''
import math

plt.axis([-100, 80, -65, 60])
for i in range(0, len(x_pos), 2):
    if i < 297:
        x1 = x_pos[i]
        y1 = y_pos[i]
        x2 = x_pos[i+2]
        y2 = y_pos[i+2]
        if math.sqrt((x1-x2)**2 + (y1-y2)**2) < 20:
            print(str(x1)+' '+str(y1))
            plt.plot([x1,x2], [y1,y2], 'go-', markersize=10)

for i in range(0, len(x_neg), 2):
    if i < 297:
        x1 = x_neg[i]
        y1 = y_neg[i]
        x2 = x_neg[i+2]
        y2 = y_neg[i+2]
        if math.sqrt((x1-x2)**2 + (y1-y2)**2) < 20:
            print(str(x1)+' '+str(y1))
            plt.plot([x1,x2], [y1,y2], 'ro-', markersize=10)

for i in range(0, len(x_unlabel), 2):
    if i < 9000:
        x1 = x_unlabel[i]
        y1 = y_unlabel[i]
        x2 = x_unlabel[i+2]
        y2 = y_unlabel[i+2]

        if x1 < 5 and y1 < 10:
            if math.sqrt((x1-x2)**2 + (y1-y2)**2) < 25:
                print(str(x1)+' '+str(y1))
                plt.plot([x1,x2], [y1,y2], 'bo-', markersize=1)
        if x2 > -5:
            if math.sqrt((x1-x2)**2 + (y1-y2)**2) < 25:
                print(str(x1)+' '+str(y1))
                plt.plot([x1,x2], [y1,y2], 'mo-', markersize=1)
plt.show()
'''
