Prototypical Networks for Few-shot Learning
Jake Snell
University of Toronto∗
Kevin Swersky
Twitter
Richard S. Zemel
University of Toronto, Vector Institute
Abstract
We propose prototypical networks for the problem of few-shot classification, where
a classifier must generalize to new classes not seen in the training set, given only
a small number of examples of each new class. Prototypical networks learn a
metric space in which classification can be performed by computing distances
to prototype representations of each class. Compared to recent approaches for
few-shot learning, they reflect a simpler inductive bias that is beneficial in this
limited-data regime, and achieve excellent results. We provide an analysis showing
that some simple design decisions can yield substantial improvements over recent
approaches involving complicated architectural choices and meta-learning. We
further extend prototypical networks to zero-shot learning and achieve state-of-theart results on the CU-Birds dataset.
1 Introduction
Few-shot classification [20, 16, 13] is a task in which a classifier must be adapted to accommodate
new classes not seen in training, given only a few examples of each of these classes. A naive approach,
such as re-training the model on the new data, would severely overfit. While the problem is quite
difficult, it has been demonstrated that humans have the ability to perform even one-shot classification,
where only a single example of each new class is given, with a high degree of accuracy [16].
Two recent approaches have made significant progress in few-shot learning. Vinyals et al. [29]
proposed matching networks, which uses an attention mechanism over a learned embedding of the
labeled set of examples (the support set) to predict classes for the unlabeled points (the query set).
Matching networks can be interpreted as a weighted nearest-neighbor classifier applied within an
embedding space. Notably, this model utilizes sampled mini-batches called episodes during training,
where each episode is designed to mimic the few-shot task by subsampling classes as well as data
points. The use of episodes makes the training problem more faithful to the test environment and
thereby improves generalization. Ravi and Larochelle [22] take the episodic training idea further
and propose a meta-learning approach to few-shot learning. Their approach involves training an
LSTM [9] to produce the updates to a classifier, given an episode, such that it will generalize well to
a test-set. Here, rather than training a single model over multiple episodes, the LSTM meta-learner
learns to train a custom model for each episode.
We attack the problem of few-shot learning by addressing the key issue of overfitting. Since data is
severely limited, we work under the assumption that a classifier should have a very simple inductive
bias. Our approach, prototypical networks, is based on the idea that there exists an embedding in
which points cluster around a single prototype representation for each class. In order to do this,
we learn a non-linear mapping of the input into an embedding space using a neural network and
take a class’s prototype to be the mean of its support set in the embedding space. Classification
is then performed for an embedded query point by simply finding the nearest class prototype. We
follow the same approach to tackle zero-shot learning; here each class comes with meta-data giving
a high-level description of the class rather than a small number of labeled examples. We therefore
learn an embedding of the meta-data into a shared space to serve as the prototype for each class.
*Initial work by first author done while at Twitter.
arXiv:1703.05175v2 [cs.LG] 19 Jun 2017
c1
c2
c3
x
(a) Few-shot
v1
v2
v3
c1
c2
c3
x
(b) Zero-shot
Figure 1: Prototypical networks in the few-shot and zero-shot scenarios. Left: Few-shot prototypes
ck are computed as the mean of embedded support examples for each class. Right: Zero-shot
prototypes ck are produced by embedding class meta-data vk. In either case, embedded query points
are classified via a softmax over distances to class prototypes: pφ(y = k|x) ∝ exp(−d(fφ(x), ck)).
Classification is performed, as in the few-shot scenario, by finding the nearest class prototype for an
embedded query point.
In this paper, we formulate prototypical networks for both the few-shot and zero-shot settings. We
draw connections to matching networks in the one-shot setting, and analyze the underlying distance
function used in the model. In particular, we relate prototypical networks to clustering [4] in order to
justify the use of class means as prototypes when distances are computed with a Bregman divergence,
such as squared Euclidean distance. We find empirically that the choice of distance is vital, as
Euclidean distance greatly outperforms the more commonly used cosine similarity. On several
benchmark tasks, we achieve state-of-the-art performance. Prototypical networks are simpler and
more efficient than recent meta-learning algorithms, making them an appealing approach to few-shot
and zero-shot learning.
2 Prototypical Networks
2.1 Notation
In few-shot classification we are given a small support set of N labeled examples S =
{(x1, y1), . . . ,(xN , yN )} where each xi ∈ R
D is the D-dimensional feature vector of an example
and yi ∈ {1, . . . , K} is the corresponding label. Sk denotes the set of examples labeled with class k.
2.2 Model
Prototypical networks compute an M-dimensional representation ck ∈ RM, or prototype, of each
class through an embedding function fφ : R
D → RM with learnable parameters φ. Each prototype
is the mean vector of the embedded support points belonging to its class:
ck =
1
|Sk|
X
(xi,yi)∈Sk
fφ(xi) (1)
Given a distance function d : RM × RM → [0, +∞), prototypical networks produce a distribution
over classes for a query point x based on a softmax over distances to the prototypes in the embedding
space:
pφ(y = k | x) = exp(−d(fφ(x), ck))
P
k0 exp(−d(fφ(x), ck0 )) (2)
Learning proceeds by minimizing the negative log-probability J(φ) = − log pφ(y = k | x) of the
true class k via SGD. Training episodes are formed by randomly selecting a subset of classes from
the training set, then choosing a subset of examples within each class to act as the support set and a
subset of the remainder to serve as query points. Pseudocode to compute the loss J(φ) for a training
episode is provided in Algorithm 1.
2
Algorithm 1 Training episode loss computation for prototypical networks. N is the number of
examples in the training set, K is the number of classes in the training set, NC ≤ K is the number
of classes per episode, NS is the number of support examples per class, NQ is the number of query
examples per class. RANDOMSAMPLE(S, N) denotes a set of N elements chosen uniformly at
random from set S, without replacement.
Input: Training set D = {(x1, y1), . . . ,(xN , yN )}, where each yi ∈ {1, . . . , K}. Dk denotes the
subset of D containing all elements (xi
, yi) such that yi = k.
Output: The loss J for a randomly generated training episode.
V ← RANDOMSAMPLE({1, . . . , K}, NC ) . Select class indices for episode
for k in {1, . . . , NC } do
Sk ← RANDOMSAMPLE(DVk
, NS) . Select support examples
Qk ← RANDOMSAMPLE(DVk
\ Sk, NQ) . Select query examples
ck ←
1
NC
X
(xi,yi)∈Sk
fφ(xi) . Compute prototype from support examples
end for
J ← 0 . Initialize loss
for k in {1, . . . , NC } do
for (x, y) in Qk do
J ← J +
1
NC NQ
"
d(fφ(x), ck)) + logX
k0
exp(−d(fφ(x), ck))#
. Update loss
end for
end for
2.3 Prototypical Networks as Mixture Density Estimation
For a particular class of distance functions, known as regular Bregman divergences [4], the prototypical networks algorithm is equivalent to performing mixture density estimation on the support set with
an exponential family density. A regular Bregman divergence dϕ is defined as:
dϕ(z, z
0
) = ϕ(z) − ϕ(z
0
) − (z − z
0
)
T ∇ϕ(z
0
), (3)
where ϕ is a differentiable, strictly convex function of the Legendre type. Examples of Bregman
divergences include squared Euclidean distance kz − z
0k
2
and Mahalanobis distance.
Prototype computation can be viewed in terms of hard clustering on the support set, with one cluster
per class and each support point assigned to its corresponding class cluster. It has been shown [4]
for Bregman divergences that the cluster representative achieving minimal distance to its assigned
points is the cluster mean. Thus the prototype computation in Equation (1) yields optimal cluster
representatives given the support set labels when a Bregman divergence is used.
Moreover, any regular exponential family distribution pψ(z|θ) with parameters θ and cumulant
function ψ can be written in terms of a uniquely determined regular Bregman divergence [4]:
pψ(z|θ) = exp{z
T θ − ψ(θ) − gψ(z)} = exp{−dϕ(z, µ(θ)) − gϕ(z)} (4)
Consider now a regular exponential family mixture model with parameters Γ = {θk, πk}
K
k=1:
p(z|Γ) = X
K
k=1
πkpψ(z|θk) = X
K
k=1
πk exp(−dϕ(z, µ(θk)) − gϕ(z)) (5)
Given Γ, inference of the cluster assignment y for an unlabeled point z becomes:
p(y = k|z) = πk exp(−dϕ(z, µ(θk)))
P
k0 πk0 exp(−dϕ(z, µ(θk))) (6)
For an equally-weighted mixture model with one cluster per class, cluster assignment inference (6) is
equivalent to query class prediction (2) with fφ(x) = z and ck = µ(θk). In this case, prototypical
networks are effectively performing mixture density estimation with an exponential family distribution
determined by dϕ. The choice of distance therefore specifies modeling assumptions about the classconditional data distribution in the embedding space.
3
2.4 Reinterpretation as a Linear Model
A simple analysis is useful in gaining insight into the nature of the learned classifier. When we use
Euclidean distance d(z, z
0
) = kz − z
0k
2
, then the model in Equation (2) is equivalent to a linear
model with a particular parameterization [19]. To see this, expand the term in the exponent:
−kfφ(x) − ckk
2 = −fφ(x)
>fφ(x) + 2c
>
k
fφ(x) − c
>
k ck (7)
The first term in Equation (7) is constant with respect to the class k, so it does not affect the softmax
probabilities. We can write the remaining terms as a linear model as follows:
2c
>
k
fφ(x) − c
>
k ck = w>
k
fφ(x) + bk, where wk = 2ck and bk = −c
>
k ck (8)
We focus primarily on squared Euclidean distance (corresponding to spherical Gaussian densities) in
this work. Our results indicate that Euclidean distance is an effective choice despite the equivalence
to a linear model. We hypothesize this is because all of the required non-linearity can be learned
within the embedding function. Indeed, this is the approach that modern neural network classification
systems currently use, e.g., [14, 28].
2.5 Comparison to Matching Networks
Prototypical networks differ from matching networks in the few-shot case with equivalence in the
one-shot scenario. Matching networks [29] produce a weighted nearest neighbor classifier given the
support set, while prototypical networks produce a linear classifier when squared Euclidean distance
is used. In the case of one-shot learning, ck = xk since there is only one support point per class, and
matching networks and prototypical networks become equivalent.
A natural question is whether it makes sense to use multiple prototypes per class instead of just one.
If the number of prototypes per class is fixed and greater than 1, then this would require a partitioning
scheme to further cluster the support points within a class. This has been proposed in Mensink
et al. [19] and Rippel et al. [25]; however both methods require a separate partitioning phase that is
decoupled from the weight updates, while our approach is simple to learn with ordinary gradient
descent methods.
Vinyals et al. [29] propose a number of extensions, including decoupling the embedding functions of
the support and query points, and using a second-level, fully-conditional embedding (FCE) that takes
into account specific points in each episode. These could likewise be incorporated into prototypical
networks, however they increase the number of learnable parameters, and FCE imposes an arbitrary
ordering on the support set using a bi-directional LSTM. Instead, we show that it is possible to
achieve the same level of performance using simple design choices, which we outline next.
2.6 Design Choices
Distance metric Vinyals et al. [29] and Ravi and Larochelle [22] apply matching networks using
cosine distance. However for both prototypical and matching networks any distance is permissible,
and we found that using squared Euclidean distance can greatly improve results for both. We
conjecture this is primarily due to cosine distance not being a Bregman divergence, and thus the
equivalence to mixture density estimation discussed in Section 2.3 does not hold.
Episode composition A straightforward way to construct episodes, used in Vinyals et al. [29] and
Ravi and Larochelle [22], is to choose Nc classes and NS support points per class in order to match
the expected situation at test-time. That is, if we expect at test-time to perform 5-way classification
and 1-shot learning, then training episodes could be comprised of Nc = 5, NS = 1. We have found,
however, that it can be extremely beneficial to train with a higher Nc, or “way”, than will be used
at test-time. In our experiments, we tune the training Nc on a held-out validation set. Another
consideration is whether to match NS, or “shot”, at train and test-time. For prototypical networks,
we found that it is usually best to train and test with the same “shot” number.
2.7 Zero-Shot Learning
Zero-shot learning differs from few-shot learning in that instead of being given a support set of
training points, we are given a class meta-data vector vk for each class. These could be determined
4
Table 1: Few-shot classification accuracies on Omniglot.
5-way Acc. 20-way Acc.
Model Dist. Fine Tune 1-shot 5-shot 1-shot 5-shot
MATCHING NETWORKS [29] Cosine N 98.1% 98.9% 93.8% 98.5%
MATCHING NETWORKS [29] Cosine Y 97.9% 98.7% 93.5% 98.7%
NEURAL STATISTICIAN [6] - N 98.1% 99.5% 93.2% 98.1%
PROTOTYPICAL NETWORKS (OURS) Euclid. N 98.8% 99.7% 96.0% 98.9%
in advance, or they could be learned from e.g., raw text [7]. Modifying prototypical networks to deal
with the zero-shot case is straightforward: we simply define ck = gϑ(vk) to be a separate embedding
of the meta-data vector. An illustration of the zero-shot procedure for prototypical networks as
it relates to the few-shot procedure is shown in Figure 1. Since the meta-data vector and query
point come from different input domains, we found it was helpful empirically to fix the prototype
embedding g to have unit length, however we do not constrain the query embedding f.
3 Experiments
For few-shot learning, we performed experiments on Omniglot [16] and the miniImageNet version
of ILSVRC-2012 [26] with the splits proposed by Ravi and Larochelle [22]. We perform zero-shot
experiments on the 2011 version of the Caltech UCSD bird dataset (CUB-200 2011) [31].
3.1 Omniglot Few-shot Classification
Omniglot [16] is a dataset of 1623 handwritten characters collected from 50 alphabets. There are 20
examples associated with each character, where each example is drawn by a different human subject.
We follow the procedure of Vinyals et al. [29] by resizing the grayscale images to 28 × 28 and
augmenting the character classes with rotations in multiples of 90 degrees. We use 1200 characters
plus rotations for training (4,800 classes in total) and the remaining classes, including rotations, for
test. Our embedding architecture mirrors that used by Vinyals et al. [29] and is composed of four
convolutional blocks. Each block comprises a 64-filter 3 × 3 convolution, batch normalization layer
[10], a ReLU nonlinearity and a 2 × 2 max-pooling layer. When applied to the 28 × 28 Omniglot
images this architecture results in a 64-dimensional output space. We use the same encoder for
embedding both support and query points. All of our models were trained via SGD with Adam [11].
We used an initial learning rate of 10−3
and cut the learning rate in half every 2000 episodes. No
regularization was used other than batch normalization.
We trained prototypical networks using Euclidean distance in the 1-shot and 5-shot scenarios with
training episodes containing 60 classes and 5 query points per class. We found that it is advantageous
to match the training-shot with the test-shot, and to use more classes (higher “way”) per training
episode rather than fewer. We compare against various baselines, including the neural statistician
[6] and both the fine-tuned and non-fine-tuned versions of matching networks [29]. We computed
classification accuracy for our models averaged over 1000 randomly generated episodes from the test
set. The results are shown in Table 1 and to our knowledge they represent the state-of-the-art on this
dataset.
3.2 miniImageNet Few-shot Classification
The miniImageNet dataset, originally proposed by Vinyals et al. [29], is derived from the larger
ILSVRC-12 dataset [26]. The splits used by Vinyals et al. [29] consist of 60,000 color images of size
84 × 84 divided into 100 classes with 600 examples each. For our experiments, we use the splits
introduced by Ravi and Larochelle [22] in order to directly compare with state-of-the-art algorithms
for few-shot learning. Their splits use a different set of 100 classes, divided into 64 training, 16
validation, and 20 test classes. We follow their procedure by training on the 64 training classes and
using the 16 validation classes for monitoring generalization performance only.
We use the same four-block embedding architecture as in our Omniglot experiments, though here
it results in a 1600-dimensional output space due to the increased size of the images. We also
5
Table 2: Few-shot classification accuracies on miniImageNet. All accuracy results are averaged over
600 test episodes and are reported with 95% confidence intervals. ∗Results reported by [22].
5-way Acc.
Model Dist. Fine Tune 1-shot 5-shot
BASELINE NEAREST NEIGHBORS∗ Cosine N 28.86 ± 0.54% 49.79 ± 0.79%
MATCHING NETWORKS [29]∗ Cosine N 43.40 ± 0.78% 51.09 ± 0.71%
MATCHING NETWORKS FCE [29]∗ Cosine N 43.56 ± 0.84% 55.31 ± 0.73%
META-LEARNER LSTM [22]∗
- N 43.44 ± 0.77% 60.60 ± 0.71%
PROTOTYPICAL NETWORKS (OURS) Euclid. N 49.42 ± 0.78% 68.20 ± 0.66%
5-way
Cosine
5-way
Euclid.
20-way
Cosine
20-way
Euclid.
1-shot
20%
30%
40%
50%
60%
70%
80%
1-shot Accuracy (5-way)
Matching / Proto. Nets
5-way
Cosine
5-way
Euclid.
20-way
Cosine
20-way
Euclid.
5-shot
20%
30%
40%
50%
60%
70%
80%
5-shot Accuracy (5-way)
Matching Nets
Proto. Nets
Figure 2: Comparison showing the effect of distance metric and number of classes per training episode
on 5-way classification accuracy for both matching and prototypical networks on miniImageNet.
The x-axis indicates configuration of the training episodes (way, distance, and shot), and the y-axis
indicates 5-way test accuracy for the corresponding shot. Error bars indicate 95% confidence intervals
as computed over 600 test episodes. Note that matching networks and prototypical networks are
identical in the 1-shot case.
use the same learning rate schedule as in our Omniglot experiments and train until validation loss
stops improving. We train using 30-way episodes for 1-shot classification and 20-way episodes for
5-shot classification. We match train shot to test shot and each class contains 15 query points per
episode. We compare to the baselines as reported by Ravi and Larochelle [22], which include a
simple nearest neighbor approach on top of features learned by a classification network on the 64
training classes. The other baselines are two non-fine-tuned variants of matching networks (both
ordinary and FCE) and the Meta-Learner LSTM. As can be seen in Table 2, prototypical networks
achieves state-of-the-art here by a wide margin.
We conducted further analysis, to determine the effect of distance metric and the number of training
classes per episode on the performance of prototypical networks and matching networks. To make
the methods comparable, we use our own implementation of matching networks that utilizes the
same embedding architecture as our prototypical networks. In Figure 2 we compare cosine vs.
Euclidean distance and 5-way vs. 20-way training episodes in the 1-shot and 5-shot scenarios, with
15 query points per class per episode. We note that 20-way achieves higher accuracy than 5-way
and conjecture that the increased difficulty of 20-way classification helps the network to generalize
better, because it forces the model to make more fine-grained decisions in the embedding space. Also,
using Euclidean distance improves performance substantially over cosine distance. This effect is even
more pronounced for prototypical networks, in which computing the class prototype as the mean of
embedded support points is more naturally suited to Euclidean distances since cosine distance is not
a Bregman divergence.
3.3 CUB Zero-shot Classification
In order to assess the suitability of our approach for zero-shot learning, we also run experiments on
the Caltech-UCSD Birds (CUB) 200-2011 dataset [31]. The CUB dataset contains 11,788 images of
200 bird species. We closely follow the procedure of Reed et al. [23] in preparing the data. We use
6
Table 3: Zero-shot classification accuracies on CUB-200.
Model Image
Features
50-way Acc.
0-shot
ALE [1] Fisher 26.9%
SJE [2] AlexNet 40.3%
SAMPLE CLUSTERING [17] AlexNet 44.3%
SJE [2] GoogLeNet 50.1%
DS-SJE [23] GoogLeNet 50.4%
DA-SJE [23] GoogLeNet 50.9%
PROTO. NETS (OURS) GoogLeNet 54.6%
their splits to divide the classes into 100 training, 50 validation, and 50 test. For images we use 1,024-
dimensional features extracted by applying GoogLeNet [28] to middle, upper left, upper right, lower
left, and lower right crops of the original and horizontally-flipped image2
. At test time we use only
the middle crop of the original image. For class meta-data we use the 312-dimensional continuous
attribute vectors provided with the CUB dataset. These attributes encode various characteristics of
the bird species such as their color, shape, and feather patterns.
We learned a simple linear mapping on top of both the 1024-dimensional image features and the
312-dimensional attribute vectors to produce a 1,024-dimensional output space. For this dataset we
found it helpful to normalize the class prototypes (embedded attribute vectors) to be of unit length,
since the attribute vectors come from a different domain than the images. Training episodes were
constructed with 50 classes and 10 query images per class. The embeddings were optimized via SGD
with Adam at a fixed learning rate of 10−4
and weight decay of 10−5
. Early stopping on validation
loss was used to determine the optimal number of epochs for retraining on the training plus validation
set.
Table 3 shows that we achieve state-of-the-art results by a large margin when compared to methods
utilizing attributes as class meta-data. We compare our method to other embedding approaches, such
as ALE [1], SJE [2], and DS-SJE/DA-SJE [23]. We also compare to a recent clustering approach
[17] which trains an SVM on a learned feature space obtained by fine-tuning AlexNet [14]. These
zero-shot classification results demonstrate that our approach is general enough to be applied even
when the data points (images) are from a different domain relative to the classes (attributes).
4 Related Work
The literature on metric learning is vast [15, 5]; we summarize here the work most relevant to our
proposed method. Neighborhood Components Analysis (NCA) [8] learns a Mahalanobis distance to
maximize K-nearest-neighbor’s (KNN) leave-one-out accuracy in the transformed space. Salakhutdinov and Hinton [27] extend NCA by using a neural network to perform the transformation. Large
margin nearest neighbor (LMNN) classification [30] also attempts to optimize KNN accuracy but
does so using a hinge loss that encourages the local neighborhood of a point to contain other points
with the same label. The DNet-KNN [21] is another margin-based method that improves upon LMNN
by utilizing a neural network to perform the embedding instead of a simple linear transformation.
Of these, our method is most similar to the non-linear extension of NCA [27] because we use a
neural network to perform the embedding and we optimize a softmax based on Euclidean distances
in the transformed space, as opposed to a margin loss. A key distinction between our approach
and non-linear NCA is that we form a softmax directly over classes, rather than individual points,
computed from distances to each class’s prototype representation. This allows each class to have a
concise representation independent of the number of data points and obviates the need to store the
entire support set to make predictions.
Our approach is also similar to the nearest class mean approach [19], where each class is represented
by the mean of its examples. This approach was developed to rapidly incorporate new classes into
a classifier without retraining, however it relies on a linear embedding and was designed to handle
2
Features downloaded from https://github.com/reedscot/cvpr2016.
7
the case where the novel classes come with a large number of examples. In contrast, our approach
utilizes neural networks to non-linearly embed points and we couple this with episodic training
in order to handle the few-shot scenario. Mensink et al. attempt to extend their approach to also
perform non-linear classification, but they do so by allowing classes to have multiple prototypes.
They find these prototypes in a pre-processing step by using k-means on the input space and then
perform a multi-modal variant of their linear embedding. Prototypical networks, on the other hand,
learn a non-linear embedding in an end-to-end manner with no such pre-processing, producing a
non-linear classifier that still only requires one prototype per class. In addition, our approach naturally
generalizes to other distance functions, particularly Bregman divergences.
Another relevant few-shot learning method is the meta-learning approach proposed in Ravi and
Larochelle [22]. The key insight here is that LSTM dynamics and gradient descent can be written
in effectively the same way. An LSTM can then be trained to itself train a model from a given
episode, with the performance goal of generalizing well on the query points. Matching networks
and prototypical networks can also be seen as forms of meta-learning, in the sense that they produce
simple classifiers dynamically from new training episodes; however the core embeddings they rely
on are fixed after training. The FCE extension to matching nets involves a secondary embedding that
depends on the support set. However, in the few-shot scenario the amount of data is so small that a
simple inductive bias seems to work well, without the need to learn a custom embedding for each
episode.
Prototypical networks are also related to the neural statistician [6] from the generative modeling
literature, which extends the variational autoencoder [12, 24] to learn generative models of datasets
rather than individual points. One component of the neural statistician is the “statistic network” which
summarizes a set of data points into a statistic vector. It does this by encoding each point within a
dataset, taking a sample mean, and applying a post-processing network to obtain an approximate
posterior over the statistic vector. Edwards and Storkey test their model for one-shot classification on
the Omniglot dataset by considering each character to be a separate dataset and making predictions
based on the class whose approximate posterior over the statistic vector has minimal KL-divergence
from the posterior inferred by the test point. Like the neural statistician, we also produce a summary
statistic for each class. However, ours is a discriminative model, as befits our discriminative task of
few-shot classification.
With respect to zero-shot learning, the use of embedded meta-data in prototypical networks resembles
the method of [3] in that both predict the weights of a linear classifier. The DS-SJE and DA-SJE
approach of [23] also learns deep multimodal embedding functions for images and class meta-data.
Unlike ours, they learn using an empirical risk loss. Neither [3] nor [23] uses episodic training, which
allows us to help speed up training and regularize the model.
5 Conclusion
We have proposed a simple method called prototypical networks for few-shot learning based on the
idea that we can represent each class by the mean of its examples in a representation space learned
by a neural network. We train these networks to specifically perform well in the few-shot setting by
using episodic training. The approach is far simpler and more efficient than recent meta-learning
approaches, and produces state-of-the-art results even without sophisticated extensions developed
for matching networks (although these can be applied to prototypical nets as well). We show how
performance can be greatly improved by carefully considering the chosen distance metric, and by
modifying the episodic learning procedure. We further demonstrate how to generalize prototypical
networks to the zero-shot setting, and achieve state-of-the-art results on the CUB-200 dataset. A
natural direction for future work is to utilize Bregman divergences other than squared Euclidean
distance, corresponding to class-conditional distributions beyond spherical Gaussians. We conducted
preliminary explorations of this, including learning a variance per dimension for each class. This did
not lead to any empirical gains, suggesting that the embedding network has enough flexibility on its
own without requiring additional fitted parameters per class. Overall, the simplicity and effectiveness
of prototypical networks makes it a promising approach for few-shot learning.
8
Acknowledgements
We would like to thank Marc Law, Sachin Ravi, Hugo Larochelle, Renjie Liao, and Oriol Vinyals
for helpful discussions. This work was supported by the Samsung GRP project and the Canadian
Institute for Advanced Research.
