# Model Results:
## 1. Label Propagation:
`labelprop_raw_algorithm.py`  
`test_raw_labelprop.py`
#### Method:
- Cross Validation
- KNN on Euclidean Distance
#### Result:
|KNN Nbr.    |  Ave. Err. | 
|:----------:|:----------:| 
|10          |   0.458    |
|30          |   0.506    |
|50          |   0.52     |
|75          |   0.498    |
|100         |   0.505    |
|150         |   0.48     |
#### Conclusion:
Useless.
#### Discussion:
One argument is that why can't we just use one labelled datapoint for each class? Won't that be enough?

## 2. Auto-encoder
`autoencoder_nn.py`  
`autoencoder_fc.py`
#### Method:
- Auto-encoder
- Transfer learning
- CNN
- Cross-Entropy
#### Result:
|NN Structure                   |  Ave. Err. |FC Structure|  Ave. Err. | 
|:-----------------------------:|:----------:|:----------:|:----------:| 
|I128-E128:64:32:24-D32:64:128  |            |            |            |
|I129-E64:32-D64:128            |            |            |            |
#### Conclusion:
Debugging
#### Discussion:

## 3. K-Means
#### Method:
- cosine distance
#### Result:
|Data Source                |  Ave. Err. | 
|--------------------------:|:----------:| 
|Old 1,000                  |    0.19    |
|new 10,000                 |    0.17    |
|new 100,000                |    0.14    |
|random 100,000             |    0.43    |
Plots are in `./result_plots` folder.
#### Conclusion:
If we have selected unlabeled data not in a random fashion, then this is not authentic unsupervised learning. 
The prior distribution of the unlabeled dataset makes a big difference in K-mean.

## 4. t-SNE
#### Method:
https://distill.pub/2016/misread-tsne/
#### Result:
Good for visualization, but where to find the class edge?
#### Conclusion:
Need further investigation
#### Discussion:

## 5. Manifold Regularization
#### Method:
#### Result:
#### Conclusion:
#### Discussion:

## 6. GAN
#### Method:
#### Result:
#### Conclusion:
#### Discussion:

