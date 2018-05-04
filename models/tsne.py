import pickle

import numpy as np
from numpy import linalg
from numpy.linalg import norm
from scipy.spatial.distance import squareform, pdist

import sklearn
from sklearn.manifold import TSNE
from sklearn.datasets import load_digits
from sklearn.preprocessing import scale

# We'll hack a bit with the t-SNE code in sklearn 0.15.2.
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.manifold.t_sne import (_joint_probabilities, _kl_divergence)

import matplotlib.pyplot as plt
import matplotlib.patheffects as PathEffects
import matplotlib

# We import seaborn to make nice plots.
import seaborn as sns
sns.set_style('darkgrid')
sns.set_palette('muted')
sns.set_context("notebook", font_scale=1.5,
                rc={"lines.linewidth": 2.5})

# import SoNYC data:
sonyc_length = 128
class_length = 1
print("** Loading SoNYC datasets **")
negative_xy = pickle.load(open("../data/negative_xy.pickle", "rb"))
positive_xy = pickle.load(open("../data/positive_xy.pickle", "rb"))

# Choose the data source
x_pool_file = "X_pool" # 1000 * 128
# x_pool_file = "X_pool_10000_new" # 10000 * 132
# x_pool_file = "X_pool_100000_new"
# x_pool_file = "X_pool_100000_random"

X_pool = pickle.load(open('../data/'+x_pool_file+'.pickle', "rb"))

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
number_iterations = 300

if use_random_state:
    tsne_result = TSNE(random_state=RS).fit_transform(full_data_x)
else:
    tsne_result = TSNE(n_components=2,
                       verbose=1,
                       perplexity=try_perplexity,
                       n_iter=number_iterations).fit_transform(full_data_x)

x_pos = tsne_result[0:300,0]
y_pos = tsne_result[0:300,1]

x_neg = tsne_result[301:600,0]
y_neg = tsne_result[301:600,1]

x_unlabel = tsne_result[601:X_pool.shape[0],0]
y_unlabel = tsne_result[601:X_pool.shape[0],1]

fig = plt.figure()
plt.scatter(x_pos,y_pos,c='g',s=50,label='pos')
plt.scatter(x_neg,y_neg,c='r',s=50,label='neg')
plt.scatter(x_unlabel,y_unlabel,c='b',s=1,label='unlabel')
plt.legend(loc='upper left')
plt.title("Preplexity:" + str(try_perplexity) + "\nIterations:" + str(number_iterations) + "\nUnlabel File:" + str(x_pool_file))
plt.savefig('result_plots/tsne_'+x_pool_file+'_p'+str(try_perplexity)+'.png', dpi=120)
plt.show()
