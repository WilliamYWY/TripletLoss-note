# TripletLoss-note

## Introduction
Triplet loss is a way to evalute and to train the networks by computing the distance between anchor point, postive point and negative point, which anchor point and positive point are in the same class.   
First, we will have to embed our inputs into vectors then compute the distance between different inputs.  
Then we back forward the loss to our network to adjust the weight and finally we expect our model to have the ability to push the negative point away from the anchor and pull the positive point toward.

## Requirements 
- Python Packages:
  - PyTorch
  - Numpy
  - Scipy

## Contents

- ### Triplet Loss
Below is the graph of the concept of triplet loss.  
  
<img src="https://user-images.githubusercontent.com/92711171/160780373-4baab362-9f90-46f8-9d1d-eaec9771a19e.png" >  
  
We expect the model to minimize the distance between "positive and anchor" and also maximize the distance between "negative and anchor" by at least **the distance between positive and anchor plus margin**.
  
Below is the equation of the Triplet loss.  
  
<img src="https://user-images.githubusercontent.com/92711171/160781985-43a296b0-7160-4cd5-90fc-a132cceaf2cf.png">
  
- f(x) takes x as an input and returns features vector.
- alpha is the margin.
- Subscript a indicates anchor point, p indicates positive point, n indicates negative point.
  
- ### Distance

There are two ways that have usually been used to compute the distance between two points: 
- Euclidean distance(L2 distance)
- Cosine distance(Cosine similarity).  

Scipy have provided a powerful function that could easily output these two distances.

```python
scipy.spatial.distance.pdist(x, "euclidean") #Euclidean distance(L2 distance)
scipy.spatial.distance.pdist(x, "cosine") #Cosine distance
```

- ### Hard Triplet loss

Below is the graph showing different kinds of triplet situations with respect to negative point at different positions.
  
<img src="https://user-images.githubusercontent.com/92711171/160787849-1c908e97-7dc3-436b-8b93-6d5ec0af439d.png" width="50%">
  
Since at many cases, the easy triplet can affect the model by quickly reducing the lost, we will have to use the hard triplet loss to back forward to the network.  
  
Which we choose:
- The maximum distance between anchor point and positive point.
- The minimum distance between anchor point and negative point.
  
Then we can input these two information into **MarginRankingLoss** to compute the loss.

```python
from torch import nn
loss = nn.MarginRankingLoss()
loss(input1, input2, y)
```
- input1 should be larger value (anchor and negative).
- input2 should be smaller value (anchor and input).
- If above two conditions matched, y = 1 and y should be a tensor with same shape as input1 and input2.
  
So in the optimal case when positive is near to anchor and negative is far away, the loss will be zero.

- ### Example
Here let's do some experiments.  
  
First we create a tensor to store feature vector for batches and also a tensor denotes the classes of those batches.  
```python
features = torch.rand([4, 5]) # 4 different points with 5 features for each
labels = torch.tensor([1,2,1,1]) # 0,2,3 are in the same class
```

Then we compute the distance between each points.  
However the function **pdist** would only return a 1D array, we need to expand the dimension to 2D to better operate the selection of hard triplet.  
```python
# L2 distance
dist_l2 = pdist(features)
dist_l2 = squareform(dist_l2)
dist_l2 = torch.from_numpy(dist_l2)
# Cosine distance
dist_cos = pdist(features, "cosine")
dist_cos = squareform(dist_cos)
dist_cos = torch.from_numpy(dist_cos)
```
dist_l2 =   
[[0.         1.10331486 0.5864828  0.59393064]  
 [1.10331486 0.         1.03922242 0.9065565 ]  
 [0.5864828  1.03922242 0.         0.80532109]  
 [0.59393064 0.9065565  0.80532109 0.        ]]  
 
dist_cos =   
 [[0.         0.42754478 0.08141503 0.10817727]  
 [0.42754478 0.         0.54301297 0.27861574]  
 [0.08141503 0.54301297 0.         0.21095896]  
 [0.10817727 0.27861574 0.21095896 0.        ]]  
  
>Note: The matrices are symmetric matrix.

Now we identify the hard triplet and store tem in to arrays.  

```python
#take L2 distance for example
n = features.shape[0]
#mask represent the relationship between points
mask = labels.expand(n,n).eq(labels.expand(n,n).t())
dist_ap, dist_an = [], []
for i in range(n):
    dist_ap.append(dist_l2[i][mask[i]].max().unsqueeze(0)) #Choose the maximum of AP
    dist_an.append(dist_l2[i][mask[i]==0].min().unsqueeze(0)) #Choose the minimum of AN

dist_ap = torch.cat(dist_ap)
dist_an = torch.cat(dist_an)
``` 
  
Then we use **MarginRankingLoss** to compute the the loss of this batch.
  
```python
y = torch.ones_like(dist_an) 
loss = torch.nn.MarginRankingLoss(0.3) # 0.3 is the margin that AP and AN should have. 
loss(dist_an, dist_ap, y) #return loss value inside a tensor
```
  
Done! Now we can feed backward the loss value to our model to continue the training.
  
## Conclusion
Hard triplet loss can train our model to achieve a higher accuracy however the training process could be very long and the model would become very hard to train if the network is too deep and have too much parameters.  
The best way would be using a pretrain CNN model such as ResNet to do the feature extraction or embedding process, then we only train the last several output dense layers using triplet loss.  
Also by using different ways to compute distance depends on different situations might also help to train our network.





