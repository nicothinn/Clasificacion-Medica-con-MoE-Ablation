Fast Deep Mixtures of Gaussian Process Experts
Clement Etienam∗ Kody Law† Sara Wade‡ Vitaly Zankin †,§
Abstract
Mixtures of experts have become an indispensable tool for flexible modelling in a supervised learning context, allowing not only the mean function but the entire density of the
output to change with the inputs. Sparse Gaussian processes (GP) have shown promise as
a leading candidate for the experts in such models, and in this article, we propose to design the gating network for selecting the experts from such mixtures of sparse GPs using a
deep neural network (DNN). Furthermore, a fast one pass algorithm called Cluster-ClassifyRegress (CCR) is leveraged to approximate the maximum a posteriori (MAP) estimator
extremely quickly. This powerful combination of model and algorithm together delivers a
novel method which is flexible, robust, and extremely efficient. In particular, the method is
able to outperform competing methods in terms of accuracy and uncertainty quantification.
The cost is competitive on low-dimensional and small data sets, but is significantly lower
for higher-dimensional and big data sets. Iteratively maximizing the distribution of experts
given allocations and allocations given experts does not provide significant improvement,
which indicates that the algorithm achieves a good approximation to the local MAP estimator very fast. This insight can be useful also in the context of other mixture of experts
models.
1 Introduction
Gaussian processes (GPs) are key components of many statistical and machine learning
models. In a Bayesian setting, they provide a probabilistic approach to model unknown
functions, which can subsequently be used to quantify uncertainty in predictions. An introduction and overview of GPs is given in (Williams and Rasmussen, 2006). In regression
tasks, the GP is a popular prior for the unknown regression function, f : x → y, due to its
nonparametric nature and tractability. It assumes that the function evaluated at any finite
set of inputs (x1, . . . , xN ) is Gaussian distributed with mean vector (µ(x1), . . . , µ(xN )) and
covariance matrix with elements K(xi, xj ), where the mean function µ(·) and the positive
semi-definite covariance (or kernel) function K(·, ·) represent the parameters of the GP.
While GPs are flexible and have been successfully applied to various problems, limitations
exist. First, GP models suffer from a high computational burden, due to the need to invert
and store large, dense covariance matrices. Second, typically parametric forms are specified
for µ(·) and K(·, ·), which crucially determine properties of the regression function, such as
spatial correlation, smoothness, and periodicity. This limits the model’s ability to recover
changing behavior of the function, e.g. different smoothness levels, across the input space.
To address the computational burden, one approach is to approximate based on multiple
GP experts. In this case, the data is partitioned into groups, and within each group, a
GP expert specifies the conditional model for the output y given the input x. Scalability is
∗NVIDIA, the work in this paper was done during employment at the University of Manchester
†University of Manchester
‡University of Edinburgh
§The Alan Turing Institute
1
arXiv:2006.13309v4 [cs.LG] 1 Dec 2023
enhanced as each expert only requires inversion of smaller matrices based on subsets of the
data. The experts’ predictions can then be combined through a product operation, known
as product of experts (PoEs, Tresp, 2000), or a sum operation, known as mixture of experts
(MoEs, Jacobs et al., 1991). The PoE approach includes the Bayesian Committee Machine
(BCM, Tresp, 2000), generalized PoE (gPoE, Cao and Fleet, 2014), and robust BCM (rBCM,
Deisenroth and Ng, 2015). PoEs are motivated by the fact that the product operation
maintains Gaussianity and are specifically designed for faster inference of stationary GP
models. However, a thorough theoretical analysis (Szabó and Van Zanten, 2019) shows that
although some PoE approaches can achieve good posterior contraction rates and asymptotic
coverage, this is only in the unrealistic non-adaptive setting. Instead, MoEs work well
even in adaptive settings and correspond to sound statistical models that take a weighted
average of the experts’ predictions. The weights are defined by the gating network, which
probabilistically maps the experts to regions of the input space. The recent work of (Trapp
et al., 2020) combines both operations to build a sum-product network of GPs.
Beyond approximation, the crucial advantage of MoEs is that model flexibility is greatly
enhanced. Specifically, any simplifying assumptions of the experts need only hold for each
subset of the data. This allows the model to infer different behaviors, such as smoothness
and variability, within different regions, and combine the multiple experts to capture nonstationary, heterogeneity, discontinuities, and multi-modality.
In this paper, our contribution is threefold. First, we construct a novel MoE model that
combines the expressive power of deep neural networks (DNNs) and the probabilistic nature
of GPs. While powerful, DNNs lack the probabilistic framework and sound uncertainty
quantification of GPs, and there has been increased interest in recent years in combining
DNNs and GPs to benefit from the advantages and overcome the limitations of each method,
see e.g. (Huang et al., 2015; Wilson et al., 2016; Iwata and Ghahramani, 2017; Daskalakis
et al., 2020) to name a few. Specifically, we use GP experts for smooth, probabilistic reconstructions of the unknown regression function within each region, while employing DNNs
for the gating network to flexibly determine the regions. To further enhance scalability, we
combine the distributed approximation of GPs through the MoE architecture with low-rank
approximations using an inducing point strategy (Snelson and Ghahramani, 2006). This
combination leads to a robust and efficient model which is able to outperform competing
models.
Second, we provide a fast and accurate approximation of the proposed deep mixture
of sparse GP experts, using a recently introduced method called Cluster-Classify-Regress
(CCR, Bernholdt et al., 2019). Finally, a novel connection is made between CCR and optimization algorithms commonly used for MAP estimation of MoEs, which lends credibility
to its success as an approximation algorithm in this context. More generally, the connection
both provides a framework to potentially further refine the CCR solution through additional
iterations of the optimization algorithm and also sheds lights on the MoE model underlying
the CCR algorithm, which provides a fast approximation for other MoE architectures as
well.
2 Methodology
For i.i.d. data (y1, . . . yN ), mixture models are an extremely useful tool for flexible density
estimation due to their ability to approximate a large class of densities and their attractive balance between smoothness and flexibility. When additional covariate information is
present and the data consists of input-output pairs, {(xi, yi)}
N
i=1, MoEs extend mixtures
to achieve flexible conditional density estimation (also known as density regression), where
the whole density of the output changes with the inputs. This is achieved by modelling the
mixture parameters as functions of the inputs, that is, by defining the gating network, which
probabilistically partitions the input space into regions, and by specifying the experts, which
characterize the relationship between x and y within each region. This results in flexible
2
framework which has been employed in numerous applications; for a recent overview, see
(Gormley and Frühwirth-Schnatter, 2019).
Specifically, the MoE model assumes that outputs are independently generated, for i =
1, . . . , N, from a mixture:
yi | xi ∼
XL
l=1
wl(xi; ψ)N

yi | f(xi; θl), σ
2
l

, (1)
where wl(·; ψ) is the gating network with parameters ψ; f(·; θl) is the regression function
for the l
th expert with parameters θl; and L is the number of experts. For simplicity, we
focus on the case when y ∈ R and employ a Gaussian likelihood for the experts, although
this may be generalized for other data types. The gating network (w1(·; ψ), . . . , wL(·; ψ)),
which maps the input space X ⊆ R
d
P
to the L−1 dimensional simplex (i.e. wl(x; ψ) ≥ 0 and
L
l=1 wl(x; ψ) = 1), reflects the relevance of each expert at any location x ∈ X .
The experts form the building blocks of the mixture model, which are combined to recover general shapes for the conditional density of y given x, that are too complicated to be
captured by a single expert. To provide intuition, MoEs can be used when the population
consists of (input-dependent) sub-populations, such that within each, a single expert provides a good description of the relationship between y and x. However, in general, the groups
may not represent actual sub-populations but rather they are combined to achieve flexibility
and approximate a wide class of conditional densities. MoEs can be augmented with a set of
allocation variables z = (z1, . . . , zN ) indicating group or cluster membership, where zi = l
if the i
th data point is generated from the l
th expert, for l = 1, . . . , L. Specifically, if
p(zi = l | xi) = wl(xi, ψ) and p(yi | xi, zi) = N

yi | f(xi; θzi
), σ
2
zi

,
then (1) is recovered after marginalization of zi:
p(yi | xi) = XL
l=1
p(yi, zi = l | xi) = XL
l=1
p(zi = l | xi)p(yi | xi, zi = l)
=
XL
l=1
wl(xi, ψ)N

yi | f(xi; θl), σ
2
l

.
Thus, letting y = (y1, . . . , yN ) and x = (x1, . . . , xN ), we can define the augmented model as
p(y, z | x) = YN
i=1
p(yi | xi, zi)p(zi | xi) = YN
i=1
wziN(yi | f(xi; θzi
), σ
2
zi
)
=
YL
l=1
Y
i:zi=l
wl(xi; ψ)N(yi | f(xi; θl), σ
2
l ).
This augmented version of the model is widely used in inference algorithms (such as expectationmaximization), as direct optimization of the likelihood in (1) is challenging, due to multimodality and well-known identifiability issues associated with mixtures. Moreover, the introduction of the allocation variables z permits distributed inference across experts.
Various formulations have been proposed in literature for both the experts and gating
networks, ranging from simple linear models to flexible nonlinear approaches. Examples
include (generalized) linear or semi-linear models (Jordan and Jacobs, 1994; Xu et al., 1995),
splines (Rigon and Durante, 2017), neural networks (Bishop, 1994; Ambrogioni et al., 2017),
GPs (Tresp, 2001; Rasmussen and Ghahramani, 2002), tree-based classifiers (Gramacy and
Lee, 2008), and others. In this work, we combine sparse GP experts with DNN gating
networks to flexibly determine regions and provide a probabilistic nonparametric model of
the unknown regression function. Figure 1 highlights the advantages of this construction:
3
1) flexible regions (over quadratic classifiers, e.g (Meeds and Osindero, 2006; Nguyen and
Bonilla, 2014; Zhang and Williamson, 2019)) and 2) well-calibrated uncertainty (over DNN
experts, e.g (Bishop, 1994; Ambrogioni et al., 2017)), through tighter credible intervals that
maintain the desired coverage.
1.00 0.75 0.50 0.25 0.00 0.25 0.50 0.75 1.00
X1
1.00
0.75
0.50
0.25
0.00
0.25
0.50
0.75
1.00
X
2
(a) True allocations
1.00 0.75 0.50 0.25 0.00 0.25 0.50 0.75 1.00
X1
1.00
0.75
0.50
0.25
0.00
0.25
0.50
0.75
1.00
X
2
(b) DNN+GP allocations
1.00 0.75 0.50 0.25 0.00 0.25 0.50 0.75 1.00
X1
1.00
0.75
0.50
0.25
0.00
0.25
0.50
0.75
1.00
X
2
(c) LR+GP allocations
1.00 0.75 0.50 0.25 0.00 0.25 0.50 0.75 1.00
X1
1.00
0.75
0.50
0.25
0.00
0.25
0.50
0.75
1.00
X
2
(d) MDN allocations
1.0 0.5 0.0 0.5 1.0
X1
1.00
0.75
0.50
0.25
0.00
0.25
0.50
0.75
1.00
X2
1.00
0.75
0.50
0.25
0.00
0.25
0.50
0.75
1.00
(e) True predictions
1.5 1.0 0.5 0.0 0.5 1.0 1.5
y
1.5
1.0
0.5
0.0
0.5
1.0
1.5
y
(f) DNN+GP predictions
1.5 1.0 0.5 0.0 0.5 1.0 1.5
y
1.5
1.0
0.5
0.0
0.5
1.0
1.5
y
(g) LR+GP predictions
1.5 1.0 0.5 0.0 0.5 1.0 1.5
y
1.5
1.0
0.5
0.0
0.5
1.0
1.5
y
(h) MDN predictions
Figure 1: Motivating toy example. Comparison of the true and estimated allocations (first row)
and predictions (second row) for the proposed DNN gating network with GP experts (second
column), logistic regression (LR) gating network with GP experts (third column), and DNN
gating network and experts (fourth column). The DNN gating network recovers the true allocations, and combined with GP experts leads to improved accuracy and uncertainty (details in
Section 4.1), especially for outlying test points.
2.1 Sparse Gaussian process experts
Mixtures of GP experts have proven to be very successful (Tresp, 2001; Rasmussen and
Ghahramani, 2002; Meeds and Osindero, 2006; Yuan and Neubauer, 2009; Nguyen and
Bonilla, 2014; Gadd et al., 2020). In particular, they overcome limitations of stationary
GPs by reducing computational complexity through distributed approximations and allowing different properties of the function for each GP expert to handle challenges, such as
discontinuities, non-stationarity, non-Gaussianity and multi-modality. They allow flexible
conditional density estimation, going beyond the iid Gaussian error assumption of standard
GP regression models.
In this case, one assumes a GP prior on the regression function for each expert with
expert-specific hyperparameters θl = (µl, ϕl):
f(·; θl) ∼ GP(µl, Kϕl
),
where µl is the mean function of the expert (for simplicity, it assumed to be constant) and ϕl
are the parameters of the covariance function Kϕl
, whose chosen form and hyperparameters
4
encapsulate properties of the function for each expert. While it is common to use zero
mean functions in standard GP models, which is made appropriate by first subtracting
the overall mean from the output, in mixtures, we must include a constant mean, as the
clustering structure is unknown and the data cannot be centred for each expert. Letting
fl,i = f(xi; θl) denote the l
th function evaluated at xi with fl = {fl,i}zi=l, the GP prior
implies a multivariate normal prior on the function fl evaluated at the inputs within the l
th
cluster:
fl ∼ N(µl, Kl,Nl
),
where µl is a vector with entries µl; Kl,Nl
represents the Nl × Nl matrix obtained by
evaluating the covariance function Kϕl
at each pair of inputs in the l
th cluster; and Nl is
number data points in the l
th cluster. With the Gaussian likelihood, the unknown function
can be marginalized, and the likelihood of the data within each cluster is:
p(yl | xl) = Z Y
i:zi=l
N(yi | fl,i, σ
2
l )N(fl | µl, Kl,Nl
)dfl
= N(yl | µl, Kl,Nl + σ
2
l INl
),
where yl and xl contain the outputs and inputs of the l
th cluster, i.e. yl = {yi}zi=l and
xl = {xi}zi=l.
GP experts are appealing due to their flexibility, intrepretability and probabilistic nature,
but they come with an increased computational cost. Specifically, given the allocation
variables z, the GP hyperparameters, which crucially determine the behavior of the unknown
function, can be estimated by optimizing the log marginal likelihood:
log(p(y | x, z)) = XL
l=1
log
N(yl | µl, Kl,Nl + σ
2
l INl
)

.
This requires inversion of Nl × Nl matrices. Compared with standard GP models, the cost
is reduced from O(N
3
) to O(
PL
l=1 N
3
l
), however this can still be expensive, depending on
the size of clusters.
To further improve scalability, one can resort to approximate methods for GPs, including
sparse GPs (Snelson and Ghahramani, 2006; Titsias, 2009; Bui et al., 2017), predictive
processes (Banerjee et al., 2008), basis function approximations (Cressie and Johannesson,
2008), or sparse formulations of the precision matrix (Lindgren et al., 2011; Grigorievskiy
et al., 2017; Durrande et al., 2019), among others (see reviews in Chp. 8 of (Williams and
Rasmussen, 2006) and (Heaton et al., 2019)). In the present work, we employ sparse GPs,
which augment the model with a set of Ml < Nl pseudo-inputs x˜l = (˜xl,1, . . . , x˜l,Ml
) and
pseudo-targets ˜fl = ( ˜fl,1, . . . ,
˜fl,Ml
) representing the l
th function evaluated at the pseudoinputs, i.e. ˜fl,m = f(˜xl,m; θl) for m = 1, . . . , Ml. Then, the key assumption for scalability
is that the prior on the regression function within each cluster factorizes given the pseudotargets:
p(fl,
˜fl | xl, x˜l) = p(fl |
˜fl, xl, x˜l)p(
˜fl | x˜l)
≈
Y
i:zi=l
p(fl,i |
˜fl, xl, x˜l)p(
˜fl | x˜l).
From properties of GPs, the prior of the pseudo-targets is ˜fl ∼ N(µl, Kl,Ml
), and the augmented model for fl given the pseudo-targets is:
p(fl | xl, x˜l,
˜fl) = Y
i:zi=l
N(fl,i | µbl,i, λl,i),
5
where
µbl,i = µl + (kl,Ml
,i)
T
(Kl,Ml
)
−1
(
˜fl − µl), (2)
λl,i = Kϕl
(xi, xi) − (kl,Ml
,i)
T
(Kl,Ml
)
−1
kl,Ml
,i, (3)
with Kl,Ml denoting the Ml × Ml matrix with elements Kϕl
(˜xl,j , x˜l,h) and kl,Ml
,i denoting
the vector of length Ml with elements Kϕl
(˜xl,j , xi). This corresponds to the fully independent training conditional (FITC) approximation (Quiñonero-Candela and Rasmussen, 2005)
(see (Bauer et al., 2016) for a further discussion of FITC). For the Gaussian likelihood, fl
can be marginalized and the likelihood of the data points within each cluster also factorizes
as:
p(yl | xl, x˜l,
˜fl) = Y
i:zi=l
N(yi | µbl,i, σ
2
l + λl,i). (4)
After marginalization of the pseudo-targets, the pseudo-inputs x˜l and hyperparameters
(µl, ϕl) can be estimated by optimizing the marginal likelihood:
log(p(y | x, z, x˜)) = XL
l=1
log (N(yl | µl, Σl)) , (5)
where Σl = (Kl,MlNl
)
T
(Kl,Ml
)
−1Kl,MlNl + Λl +σ
2
l INl
; Kl,MlNl
is the Ml × Nl matrix with
columns kl,Ml
,i; and Λl is the diagonal matrix with diagonal entries λl,i. This strategy
allows us to reduce the complexity to O(
PL
l=1 NlM2
l
).
2.2 Deep neural gating networks
The choice of gating network is crucial to flexibly determine the regions and accurately approximate the conditional densities. Indeed, various proposals exist to combine GP experts
with different gating networks, and approaches to specify the gating network can be divided
into generative or discriminative classifiers. Generative models specify a joint mixture model
for y and x and typically assume a Gaussian distribution for the inputs within each class,
resulting in linear or quadratic discriminant analysis, e.g. (Meeds and Osindero, 2006; Chen
et al., 2014; Zhang and Williamson, 2019). To allow more flexibility in the regions, (Yuan
and Neubauer, 2009; Gadd et al., 2020) extend this by considering a mixture of Gaussians
within each class. Discriminative models, on the other hand, avoid modelling the inputs and
focus on the conditional density of interest by defining the gating network directly. Discriminative models include linear classifiers (Nguyen and Bonilla, 2014) and tree-based classifiers
(Gramacy and Lee, 2008), which may result in rigid assumptions on the partition structure
of the input space. The latter, for example, assumes axis-aligned rectangular regions, and
while more flexible partitioning approaches exist, such as Voronoi tessellations (Pope et al.,
2019), they come at an increased cost. Other flexible proposals include GP classifiers (Tresp,
2001) and kernel-based methods (Rasmussen and Ghahramani, 2002), but these similarly
suffer from the trade-off between flexibility and cost.
In this work, we focus on DNNs, which are known to be universal for classification
(Szymanski and McCane, 2012). DNNs flexibly determine the regions, removing any rigid
assumptions, without increasing the computational cost. Specifically, we define the gating
network by a feedforward DNN with a softmax output:
wl(x; ψ) = exp(hl(x; ψ))
PL
j=1 exp(hj (x; ψ))
,
where hl is the l
th component of h : R
d → R
L
, defined by
h( · ; ψ) = ηJ (ηJ−1(· · · η1( · ; ψ1)· · · ; ψJ−1); ψJ ),
6
with ηj : R
dj−1 → R
dj (d0 = d, dJ = L) the j
th layer of a neural network
ηj ( · ; ψj ) : x 7→ ηj (x; ψj ) = ReLU(Ajx + bj ),
where ReLU(x) = max{0, x} is the element-wise rectifier; ψj = {Aj , bj} comprises the
weights Aj ∈ R
dj×dj−1 and biases bj ∈ R
dj
for level j = 1, . . . , J; and ψ = (ψ1, . . . , ψJ )
collects all the parameters of the DNN.
Deep neural gating networks have been used in literature but are typically combined
with DNN experts (Bishop, 1994; Ambrogioni et al., 2017). The mixture density network
(MDN, Bishop, 1994) uses this gating network but parametrizes both the regression function and variance of the Gaussian model in (1) by DNNs. This offers considerable flexibility
beyond standard DNN regression, but significant valuable information can be gained with
GP experts. Specifically, as the number of data points in each cluster is data-driven, DNN
experts may overfit due to small cluster sizes. In addition, DNNs are susceptible to adversarial attacks and tend to be overconfident even when predictions are incorrect (Szegedy
et al., 2013). Moreover, DNNs may underestimate variability, especially for test data which
are quite different from the observed data (Nguyen et al., 2015) (see Figure 1). Instead,
GP experts probabilistically model the local regression function, avoiding overfitting and
providing well-calibrated uncertainty quantification.
3 Inference
While Markov chain Monte Carlo algorithms provide full Bayesian inference for MoEs, they
rely on expensive Gibbs samplers (Rasmussen and Ghahramani, 2002; Meeds and Osindero,
2006; Gadd et al., 2020), which alternately sample the allocation variables z given the model
parameters (that is, the gating and experts parameters) and the model parameters given
z. To alleviate the cost, (Zhang and Williamson, 2019) use importance sampling and parallelization. However, accurate estimation requires the number of important samples to be
exponential in the Kullback-Leibler divergence between the proposal and posterior (Chatterjee and Diaconis, 2018); this can be prohibitive for MoEs due to the massive dimension
of the partition space. A popular alternative strategy is approximate inference, where the
Gibbs sampling steps for allocation variables and the model parameters are replaced, respectively, with either an expectation or maximization step (Kurihara and Welling, 2009).
An expectation-expectation strategy corresponds to mean-field variational Bayes for MoEs
(Yuan and Neubauer, 2009). While popular expectation-maximization (EM) algorithms can
be used for maximum a posteriori (MAP) estimation as in (Tresp, 2001), we instead focus
on the faster maximization-maximization (MM) strategy. For generative mixtures of GP
experts, an MM algorithm was developed in (Chen et al., 2014).
Our MM algorithm provides MAP estimation for the augmented model by optimising
the augmented posterior:
π(ψ, θ, σ2
, z,
˜f | y, x, x˜) ∝ p(y, z,
˜f | x, x˜, ψ, θ, σ2
)π(ψ, θ, σ2
)
=
YL
l=1


Y
i:zi=l
wl(xi; ψ)N(yi | µbl,i, σ
2
l + λl,i)

 N(
˜fl | µl, Kl,Ml
)π(ψ, θ, σ2
),
where θ = (θ1, . . . , θL), σ
2 = (σ
2
1, . . . , σ2
L), and ˜f = (˜f1, . . . ,
˜fL) collect the expert-specific
GP hyperparameters, noise variance, and pseudo-targets, respectively. In the following, we
consider vague priors π(ψ, θ, σ2
) ∝ 1, and thus equivalently optimise the augmented log
7
posterior:
log(π(ψ, θ, σ2
, z,
˜f | y, x, x˜)) = const. +
XL
l=1
log(N(
˜fl | µl, Kl,Ml
))
+
XL
l=1
X
i:zi=l
log(wl(xi; ψ)) + log(N(yi | µbl,i, σ
2
l + λl,i)).
For GP experts, this provides maximum marginal likelihood estimation or type II MLE
of the GP hyperparameters and noise variance (θl, σ2
l
), as the GP regression functions are
marginalized. For sparse GP experts, we have the additional parameters (x˜l,
˜fl). The
pseudo-inputs x˜l are treated as hyperparameters to be optimized, and while pseudo-targets ˜fl
can be analytically integrated, we also estimate ˜fl with its posterior mean for faster inference
(see Section 3.1 for further details). The MM algorithm is an iterative conditional modes
algorithm (Kittler and Föglein, 1984) (can also be viewed as a coordinate ascent method)
that alternates between optimizing the allocation variables z and the model parameters. It
is guaranteed to never decrease the log posterior of the augmented model and therefore will
converge to a fixed point (Kurihara and Welling, 2009). However, it is susceptible to local
maxima, which can be alleviated with multiple restarts of random initializations.
3.1 Maximization-maximization
Our MM algorithm is defined by iterating over the following two steps: optimize the (i)
latent cluster allocation variables and (ii) gating and expert parameters
(i) z = argmax
z∈{1,...,L}N
log π(z | ψ, θ, σ2
,
˜f, x˜, x, y),
(ii) (ψ, θ, σ2
,
˜f, x˜) = argmax
(ψ,θ,σ2,˜f ,x˜)
log π(ψ, θ, σ2
,
˜f | x˜, z, x, y).
(6)
We note that while the pseudo-targets ˜f can be marginalized, the full conditional for z
in the first step of (6) would be:
π(z | ψ, θ, σ2
, x˜, x, y) = Z
π(z,
˜f | ψ, θ, σ2
, x˜, x, y)d˜f
∝ p(z | ψ, x)p(y | x, z, x˜, θ, σ2
),
with optimal allocation:
z = argmax
z∈{1,...,L}N
log(π(z | ψ, θ, σ2
, x˜, x, y))
= argmax
z∈{1,...,L}N
XN
i=1
log(wzi
(xi; ψ)) +XL
l=1
log (N(yl | µl, Σl)) . (7)
Recall from (5) that Σl is a full covariance matrix; thus, the second term in (7) does not
factorize across i = 1, . . . , N and direct optimization over the space {1, . . . , L}
N is infeasible.
An iterative conditional modes algorithm could be employed which cycles over each data
point i = 1, . . . , N, optimizing zi based on the conditional Gaussian likelihood given the data
points currently allocated to each cluster. For each data point, the conditional Gaussian
likelihood p(yi | zi = l, z−i, y−i) must be computed (where z−i and y−i denote z and y with
the i
th element removed), requiring rank one updates to the inverse covariance matrices for
every l = 1, . . . , L.
8
However, we can significantly reduce the computational cost by also estimating ˜f. In
this case, the full conditional for z in (i) of (6) factorizes across i = 1, . . . , N:
π(z | ψ, θ, σ2
,
˜f, x˜, x, y) ∝ p(z | ψ, x)p(y | x, z, x˜,
˜f, θ, σ2
)
=
YN
i=1
wzi
(xi; ψ)N(yi | µbzi,i, σ
2
zi + λzi,i).
Thus, the allocation can be done in parallel across the N data points:
zi = argmax
zi∈{1,...,L}
log(wzi
(xi; ψ)) + log(N(yi | µbzi,i, σ
2
zi + λzi,i)).
(8)
In the second step of the MM algorithm in (6), we perform two sub-steps, first optimizing
the gating network and expert parameters with ˜f marginalized, and then, given those optimal
values, estimating ˜f:
(a) (ψ, θ, σ2
, x˜) = argmax
(ψ,θ,σ2,x˜)
log π(ψ, θ, σ2
| x˜, z, x, y),
(b) ˜f = argmax
˜f
log π(
˜f | x˜, z, x, y, ψ, θ, σ2
).
(9)
In (a) of (9), the full conditional of (ψ, θ, σ2
) is:
π(ψ, θ, σ2
| x˜, z, x, y) ∝ p(z | x, ψ)p(y | x, x˜, z,
˜f, θ, σ2
)
=
YL
l=1


Y
i:zi=l
wl(xi; ψ)

 N(yl | µl, Σl).
Thus, optimization of the gating network and expert parameters can be done in parallel,
both between each other as well as across l = 1, . . . , L for the experts. Specifically, the
optimal gating network are:
ψ = argmax
ψ∈Ψ
XL
l=1
X
i:zi=l
hl(xi; ψ) −
XN
i=1
log XL
l=1
exp(hl(xi; ψ))!
, (10)
and expert parameters (GP hyperparameters, noise variance, and pseudo-inputs) are estimated by optimizing the log marginal likelihood in (5):
(θl, σ
2
l
, x˜l) = argmax
θ∈Θ,σ2∈R+,x˜∈RMl×d
log (N(yl | µl, Σl)) .
(11)
Then, in (b) of (9), the full conditional for ˜fl is:
π(
˜fl | x˜l, x, y, θ, σ2
) ∝ p(
˜fl | x˜l, θ, σ2
)p(y |
˜fl, x˜l, x, y, θ, σ2
)
= N(
˜fl | µl, Kl,Ml
)
Y
i:zi=l
N(yi | µbl,i, σ
2
l + λl,i),
which, following standard derivations, is shown to be Gaussian with covariance matrix
Kl,Ml
(Ql,Ml
)
−1Kl,Ml
and mean:
E[
˜fl | θl, σ
2
l
, x˜l, yl, xl]
= µl + Kl,Ml
(Ql,Ml
)
−1

Kl,MlNl
(Λl + σ
2
l INl
)
−1
(yl − µl)

, (12)
where Ql,Ml = Kl,Ml + Kl,MlNl
(Λl + σ
2
l INl
)
−1
(Kl,MlNl
)
T
. Thus, the optimal ˜fl in (b) of
(8) is the posterior mean in (12).
9
3.2 A fast approximation: CCR
The MM algorithm iterates between clustering in (8) and in parallel classification in
(10) and regression in (11) and (12). This closely resembles the CCR algorithm recently
introduced in (Bernholdt et al., 2019). The important differences are that 1) CCR is a
one pass algorithm that does not iterate between the steps and 2) CCR approximates the
clustering in the first step of the MM algorithm by a) careful re-scaling the data to emphasize
the output y in relation to x and b) subsequently applying a fast clustering algorithm, e.g.
K-means (Hartigan and Wong, 1979), Gaussian mixture model (GMM, McLachlan and
Basford, 1988), or DB-scan (Ester et al., 1996), to jointly cluster the rescaled (y, x). We
also note that the original formulation of CCR performs an additional clustering step so that
the allocation variables used by the regression correspond to the prediction of the classifier;
this is equivalent to the clustering step of the MM algorithm in (8) including only the term
associated to the gating network.
This novel connection sheds light on the MoE model underlying the CCR algorithm and
allows us to view CCR as a fast, one-pass approximation to the MM algorithm for MoEs.
Therefore, we can use CCR to construct a fast approximation of the proposed deep mixture
of sparse GP experts. Moreover, this connection can be used to obtain a fast approximation
to other discriminative MoE architectures. As shown in the following section, CCR provides
a good, fast approximation for many numerical examples. If extra computational resources
are available, the CCR solution can be improved through additional MM iterations (i.e.
it provides a good initialization for the MM algorithm). However, in our examples, we
notice that the potential for further improvement is limited. We find that MM with random
initialization can also produce a reasonable estimate in many examples after two iterations;
we refer to this algorithm as MM2r. It is fast, but we will see that it takes approximately
2-3 times longer than CCR.
3.3 Complexity considerations
Suppose we parameterize the DNN with pc parameters, and each of the sparse GP experts
is approximated with Ml pseudo-inputs. The MM algorithm described in Section 3.1 incurs a cost per iteration of clustering the N points given the current set of parameters (6).
This cost is O(N
PL
l=1 M2
l
), and it is parallel in N. The algorithm also incurs a cost per
iteration of classification with N points and L regressions using N1, . . . , NL points, where
N =
PL
l=1 Nl in (6). These operations can also be done in parallel. The cost for the classification is O(N pc) assuming that the number of epochs for training is O(1). The cost for the
regressions is O(
PL
l=1 M2
l Nl). Hence the total cost for (6) is O(N pc +
PL
l=1 M2
l Nl), which
can be roughly bounded by O(NPmax), where Pmax = max{pc, M2
1 , . . . , M2
L}. Randomly
initialized MM cannot be expected to provide reasonable results after one pass, however with
sparse GP models the first iteration provides a significant improvement which is also sometimes reasonable. Ignoring parallel considerations, the total cost for 2-pass MM (MM2r) is
O(2N pc +
PL
l=1 M2
l
(2Nl + N)).
For CCR, the cost is the same for the second step (6), while the first step is replaced
with a chosen clustering algorithm, e.g. GMM, which incurs a cost of O(NL). The latter
iterates between steps which can be parallelized in different ways. The total cost of CCR is
hence O(N(pc + L) + PL
l=1 M2
l Nl) = O(NPmax). This is a one pass algorithm, which often
provides acceptable results. The overhead for MM2r vs. CCR for our model is then roughly
O((Pmax − L)N). More precisely it is O(N(pc − L) + PL
l=1 M2
l
(N + Nl)).
10
3.4 Prediction
There are two approaches that can be employed to predict y
∗
at a test value x
∗
. Hard
allocation based prediction is based on the single best regression/expert:
z
∗ = argmax
z∗∈{1,...,L}
log(wz∗ (x
∗
; ψ)), y
∗ = µbz∗ (x
∗
), (13)
where (similiar to (2)), the GP expert’s prediction is:
µbz∗ (x
∗
) = µl + (kl,Ml
,x∗ )
T
(Kl,Ml
)
−1
(
˜fl − µl),
with kl,Ml
,x∗ denoting the vector of length Ml with elements Kϕl
(˜xl,j , x∗
). Whereas soft
allocation based prediction is given by a weighted average
y
∗ =
XL
l=1
wl(x
∗
; ψ)µbl(x
∗
). (14)
Soft-allocation may be preferred in cases when there is not a clear jump in the unknown
function, thus allowing us to smooth the predictions in regions where the classifier is unsure.
Similarly, the variance or density of the output or regression function can also be computed
at any test location to obtain measures of uncertainty in our predictions.
Hard allocation based density estimation delivers a Gaussian approximation for a
test value x
∗
,
pb(y | x
∗
, z
∗ = l) = N

y | µbl(x
∗
), σb
2
l (x
∗
)

,
with allocation z
∗
as in (13), and where σb
2
l
(x
∗
) = λl,x∗ + σ
2
l + (kl,Ml
,x∗ )
T
(Ql,Ml
)
−1kl,Ml
,x∗
and λl,x∗ is computed as in (3) at x
∗
. Soft allocation based density estimation delivers
a mixture of Gaussians for a test value x
∗
, similar to (14):
pb(y | x
∗
) = XL
l=1
wl(x
∗
; ψ)N(y | µbl(x
∗
), σb
2
l (x
∗
)).
In cases when the density of the output may be multi-modal, looking at point predictions
alone is not useful. In this setting, the soft density estimates are preferred, allowing one to
capture and visualise the multi-modality.
4 Numerical Experiments
In this section, we perform a range of experiments to highlight the flexibility of our model
and compare the accuracy and speed of the MM and CCR algorithms. For the sparse
GP experts, we use the isotropic squared exponential covariance function with a variable
number of inducing points Ml based on the cluster sizes (Burt et al., 2020; Nieman et al.,
2021). The pseudo-inputs x˜
l
are initialized via K-means and the GP hyperparameters are
initialized based on the scale of the data. The number of experts L is determined apriori
using the Bayesian information criterion (BIC) to compare the GMM clustering solutions of
the rescaled (y, x) across different values of L. The choice of GMM is motivated by allowing
elliptically-shaped clusters in the rescaled space, while keeping cost low (e.g. if compared to
the K-means). In the Appendix A, we instead consider using cross-validation and examine
the log-likelihood on the held-out data to select the number of components, but we found
that this significantly increases run time compared to BIC, with similar performance metrics.
A table reporting the number of experts can also be found in Appendix A. The architecture
of the DNN gating network is chosen with the use of Auto-Keras (Jin et al., 2019) and
a quadratic regularization is used with a value of 0.0001 for the penalization parameter.
The adaptive stochastic gradient descent solver Adam (Kingma and Ba, 2014) is used to
11
optimize the model weights and biases, with a validation fraction of 0.1 and a maximum of
500 epochs. The GPy package (GPy, 2012) and Keras (Chollet et al., 2015) were used to
train the experts and gating network respectively.
For each experiment we performed 5-fold cross-validation with random shuffling of the
data. The experiments are executed on Intel Core i7 (I7-7820HQ) with 4 cores and with
16GB of RAM.
4.1 Simulations
As a motivating toy example, we generate inputs within each cluster l = 1, 2 from a noisy
2D spiral and outputs y = (2l − 1)(x
2
1 + x
2
2) + ϵ. Figure 1 demonstrates that flexible DNN
gating networks, over quadratic classifiers (e.g. logistic regression (LR)), are required to
recover the true allocations. When combined with GP experts, this leads to improved
accuracy (R
2 = 98.74%, 86.98%, 97.04% for DNN+GP, LR+GP, MDN, respectively) and
tighter credible intervals (CI) that maintain the desired coverage (average CI length =
0.25, 1.36, 0.41 and empirical coverage (EC) = 95%, 100%, 98.7% for DNN+GP, LR+GP,
MDN, respectively). In particular, for outlying test points (red squares in Figure 1), MDN
provides highly overconfident inaccurate predictions, whereas predictions and CIs are more
accurate and realistic for the proposed DNN+GP.
4.2 Datasets
Our experiments range from small to large datasets of varying dimensions and complexity and mainly aim to highlight the model’s capability to flexibly estimate the conditional
density and capture issues such as discontinuity, non-stationarity, heteroscedasticity, and
multi-modality that challenge standard GP models. First, the Motorcycle dataset (Silverman, 1985) consists of N = 133 measurements with d = 1. Second, the NASA dataset
(Gramacy and Lee, 2008) comes from a computer simulator of a NASA rocket booster vehicle with N = 3167; we focus on modelling the lift force as a function of the speed (mach),
the angle of attack (alpha), and the slide-slip angle (beta), i.e. d = 3. Our third experiment
is the Higdon function (Higdon, 2002; Gramacy and Lee, 2008); X = [0, 20] and N = 1000.
Next, N = 10, 000 points are generated from the Bernholdt function (Bernholdt et al., 2019);
in this case, X = [−4, 10]2
and f(x) = g(x1)g(x2), where g(x) is the piece-wise smooth function studied in (Monterrubio-Gómez et al., 2020). The kin40k dataset (Seeger et al., 2003)
is a popular benchmark for GP regression methods and consists of N = 40, 000 points that
describe the location of a robotic arm as a highly nonlinear function of control input with
d = 8.
4.2.1 χ dataset
A tokamak is a device which uses magnetic fields to confine hot plasma in the shape of a
torus. It is the leading candidate for production of controlled thermonuclear power, for use
in a prospective future fusion reactor (Greenwald, 2016).
One dimensional radial transport modeling (Park et al., 2017) using theory-based models
such as GLF23 (Waltz et al., 1997), MMM95 (Bateman et al., 1998), and TGLF (Staebler
et al., 2007) plays an essential role in interpreting experimental data and guiding new experiments for magnetically confined plasmas in tokamaks. Turbulent transport resulting
from micro-instabilities have a strong nonlinear dependency on the temperature and density
gradients. One of the key characteristics is a sharp increase of turbulent flux as the gradient
of temperature increases beyond a certain critical value. This leads to a highly nonlinear
and discontinuous function of the inputs.
Here we consider an analytical stiff transport model that describes turbulent ion energy
12
transport in tokamak plasmas (Bernholdt et al., 2019; Janeschitz et al., 2002):
χ = S(RT′
/T − (RT′
/T)crit)
αH




(RT′
/T)
(RT′/T)crit



 − 1

, (15)
where χ is ion thermal diffusivity, H(·) is the Heaviside function, R is the major radius, and
T
′
is the radial derivative of ion temperature. The normalized critical gradient (RT′
/T)crit
of ion temperature is calculated using IFS/PPPL model (Kotschenreuther et al., 1995),
which is a nonlinear function of electron density (ne), electron and ion temperatures (Te, T),
safety factor (q), magnetic shear (sˆ), effective charge (Zeff ) and the normalized gradient of
ion temperature (RT′
/T) and density (Rn′
/n). It is assumed that S = 1 and α = 1.
Considering (T, T′
) and (n, n′
) as two input parameters each, this gives a total of 10 inputs
x ∈ R
10. The output (15) is y ∈ R+. Basic primitive model inputs are chosen uniformly
at random from a hypercube ω ∈ [0, 1]17, which then give rise to realistic inputs x ∈ R
10
,
which concentrate on a manifold in the ambient space. See (Kotschenreuther et al., 1995)
for details of the model, and Figure 2 for visualization of the input distribution histogram.
The data consists of N = 150, 000 simulations from (15). This model presents a challenge for the competing MoE methods, due to input dimensionality and data size, and for
competing PoE methods, due to the sharp nonlinearities and discontinuities. It therefore
serves to highlight the value of our method. Results appear in the bottom row of the tables.
0
50
100
150
200
250
300
1
0
50
100
150
200
250
300
350
2
1.4
1.6
1.8
2.0
2.2
2.4
2.6
3
1.0
1.5
2.0
2.5
3.0
4
0.0
0.1
0.2
0.3
0.4
5
0
1
2
3
4
5
6
6
0
5
10
15
7
0.2
0.3
0.4
0.5
0.6
0.7
8
0 5 10 15
0
3
4
5
6
7
8
9
9
0 100 200 300
1
0 100 200 300
2
1.5 2.0 2.5
3
1.0 1.5 2.0 2.5 3.0
4
0.0 0.2 0.4
5
0 2 4 6
6
0 5 10 15
7
0.2 0.4 0.6
8
4 6 8
9
Figure 2: Input distribution marginals of a χ dataset
13
Table 1: R2
test accuracy (%) on the five datasets for our model with CCR, CCR-MM, and
MM2r algorithms and MDN, gPoE, RBCM, and FastGP benchmarks. Results obtained with
5-fold cross-validation (standard deviation in brackets).
Model CCR CCR-MM MM2r MDN gPoE RBCM FastGP BART ORTHNAT PPGPR DSPP Deep GP Motorcycle 76.70 (9.24) 77.84 (9.87) 74.60 (12.03) 71.27 (10.84) 74.46 (8.12) 78.46 (8.42) 74.94 (10.32) 75.12 (8.65) 78.18 (10.39) 77.27 (9.27) 75.82 (11.51) 74.09 (10.04) NASA 97.07 (1.39) 96.94 (1.50) 95.75 (1.49) 96.80 (1.82) 93.97 (2.01) 94.52 (1.13) 94.71 (1.42) 96.72 (1.93) 95.96 (1.70) 96.85 (1.22) 94.82 (1.95) 96.74 (1.62) Higdon 99.91 (0.03) 99.96 (0.01) 99.89 (0.02) 99.35 (0.02) 99.90 (0.02) 99.94 (0.02) 99.87 (0.02) 99.52 (0.03) 99.44 (0.03) 99.71 (0.04) 99.45 (0.05) 99.94 (0.02) Bernholdt 99.50 (0.40) 98.00 (0.58) 94.81 (0.93) 93.34 (3.05) 89.69 (2.23) 94.71 (2.15) 95.90 (0.29) 94.17 (0.62) 91.25 (1.06) 90.98 (1.78) 96.39 (1.08) 97.12 (0.68)
kin40k 94.53 (1.21) 95.02 (1.32) 92.38 (2.03) 90.75 (2.85) 92.08 (1.97) 91.36 (2.41) 92.94 (1.78) 94.72 (1.99) 93.91 (2.13) 93.06 (2.41) 91.62 (2.12) 91.68 (2.06)
χ150k 95.71 (0.92) 97.53 (1.29) 93.62 (2.74) 89.90 (4.84) 91.92 (1.18) 90.71 (9.87) 92.99 (2.42) 91.47 (2.81) 92.38 (4.38) 94.44 (3.85) 92.04 (1.57) 92.75 (2.91)
Table 2: Mean wall-clock time (in seconds) on the five datasets for our model with CCR, CCRMM, and MM2r algorithms and MDN, gPoE, RBCM, and FastGP benchmarks. Results obtained
with 5-fold cross-validation.
Model CCR CCR-MM MM2r MDN gPoE RBCM FastGP BART ORTHNAT PPGPR DSPP DeepGP
Motorcycle 4.7 36.3 11.2 21.6 4.2 5.5 7.8 8.7 6.9 4.3 5.3 10.9
NASA 10.1 25.8 20.4 188.2 9.4 8.5 10.3 27.3 10.0 27.4 23.2 10.4
Higdon 9.7 46.8 22.9 82.1 7.2 6.9 7.7 14.7 13.6 11.5 8.1 9.5
Bernholdt 66.3 232.6 159.5 252.7 65.5 77 69.1 77.1 67.2 80.4 84.2 79.6
kin40k 85.7 301.4 179.2 284.9 92.3 127.1 120.8 107.1 119.4 116.9 101.3 91.6
χ150k 495.7 1290.1 1003.5 1886.4 1711.9 1542.5 1170.9 1650.8 1076.9 1475.3 1620.4 1511.5
4.3 Results
For our model, we compare the MM algorithm with the fast CCR and MM2r approximations. The MM algorithm is initialized at the CCR solution and iterates until the reduction
in the R
2
is below a threshold of 0.0001 or a maximum number of iterations is reached.
Competing models are two product of GP experts models, gPoE (Cao and Fleet, 2014),
RBCM (Deisenroth and Ng, 2015), a mixture of GP experts, FastGP (Nguyen and Bonilla,
2014), and MDN (Bishop, 1994), as well as a Bayesian treed-based model (BART Chipman
et al., 2010). Treed GP models (Gramacy and Lee, 2008) were also considered, although
the cost was so much larger that the method is not competitive (see Table 4). We also
report results for other methods, including ORTHNAT (Salimbeni et al., 2018), PPGPR
(Jankowiak et al., 2020b), DSPP (Jankowiak et al., 2020a), and Deep GPs (Damianou and
Lawrence, 2013).
First, we observe that CCR has similar or improved test accuracy compared to MM2r
(Table 1), and CCR reduces wall-clock time (Table 2) by a factor of 2-3 for all experiments. The CCR solution can be further refined through MM iterations (Table 1); however
this comes at a higher computational cost. Moreover, in practice we observe little to no
improvement in accuracy with further MM iterations, across all datasets.
Compared with state-of-the-art GP, neural network, and tree-based benchmarks, our
model has the highest test accuracy in almost all the experiments considered; the one exception is RBCM for Motorcycle, where the accuracy is slightly higher, but well within the
standard deviation. For the smaller datasets, CCR is slightly slower than the PoE models
(gPoE and RBCM) and FastGP, however this is offset by the improved accuracy. CCR is
substantially faster than all competitors for the χ model, with d = 10 and (relatively) big
data N = 150, 000.
Lastly, we highlight that our model provides uncertainty quantification (as well as density
estimation which can capture multi-modality in the predictions) for both the unknown
function and predictions. Figure 3 displays the soft and hard allocation predictions together
with the respective 2-σ CIs for the Motorcycle dataset. The model is able to recover the
apparent non-stationary and heteroscedasticity in the data, while the other models (similar
plots provided in the Figure 5) tend to produce intervals that are too wide (especially on
the left for x ≤ −0.6 for PoEs) or too tight. For our model, the data (in red) is contained
within the region bounded with dashed lines, suggesting that the model also provides good
empirical coverage. Table 3 gives more insights into the empirical coverage (i.e. fraction of
14
Table 3: Empirical coverage (EC95) and average length of 95% CIs (CI¯
95), averaged over 5-
folds. We highlighted the model that has the shortest average length of 95% CIs while providing
empirical coverage ≥ 95%.
Model CCR MDN gPoE RBCM FastGP BART ORTHNAT PPGPR DSPP DeepGP
EC95 CI¯
95 EC95 CI¯
95 EC95 CI¯
95 EC95 CI¯
95 EC95 CI¯
95 EC95 CI¯
95 EC95 CI¯
95 EC95 CI¯
95 EC95 CI¯
95 EC95 CI¯
95 Motorcycle 96.35 0.84 84.75 0.67 98.78 1.18 96.29 0.97 99.46 1.31 95.17 1.02 98.22 1.15 94.23 1.01 98.70 0.72 96.86 1.19
NASA 98.38 0.35 96.94 0.39 97.92 0.58 88.38 0.43 97.89 0.49 97.31 0.54 96.08 0.50 97.00 0.54 97.14 0.43 95.85 0.51
Higdon 97.32 0.06 95.90 0.09 98.60 0.10 71.70 0.08 99.90 0.09 96.79 0.07 97.36 0.12 94.56 0.08 96.35 0.07 94.75 0.11
Bernholdt 98.34 0.29 95.35 0.39 96.73 0.43 92.19 0.41 98.14 0.49 94.24 0.35 97.07 0.57 98.15 0.46 94.76 0.51 96.58 0.48
kin40k 95.12 0.51 94.89 0.56 93.67 0.61 93.23 0.58 94.40 0.66 96.25 0.63 96.24 0.73 94.92 0.67 95.09 0.62 94.30 0.67
χ150k 97.51 0.63 96.89 0.72 96.44 0.74 79.27 0.68 94.70 0.71 95.69 0.87 97.85 0.88 93.65 0.66 94.33 0.86 92.36 0.63
1.00 0.75 0.50 0.25 0.00 0.25 0.50 0.75 1.00
X
1.5
1.0
0.5
0.0
0.5
1.0
1.5
Estimate
Data
Hard
Soft
2 -CI (Soft)
2 -CI (Hard)
(a) Heat map of the conditional density
1.5 1.0 0.5 0.0 0.5 1.0
y
0.0
0.2
0.4
0.6
0.8
1.0
1.2
1.4
Density
Soft, y
* = 0.413
Hard, y
* = 0.277
Truth, y = 0.397
(b) Density estimate given x
∗ = −0.478.
Figure 3: (a) Predictions (based on soft and hard allocations) with our model for the Motorcycle
dataset, with two standard deviations and soft allocation based density estimates; (b) a slice
representing the density estimate given x
∗ = −0.478.
times the test points are contained within the 95% CIs) against the average length of the
CIs. It can be seen that the proposed model achieves tight intervals at the desired coverage,
whereas most of the other models produce overconfident predictions or conservative intervals
that are unnecessarily wide.
Table 4 shows the metrics obtained for the treed GP model on all datasets but kin40k
and χ150k because of the computational infeasibility (the treed GP package employs MCMC
inference), and hence it was separated in the standalone table. Although the treed GP
model has the advantage in terms of R
2
and empirical coverage/interval length over CCR
in several instances, it is offset by prohibitively longer execution times.
Figure 5 compares heat maps of the conditional density for Motorcycle dataset. It can be
seen that, unlike the other models, the proposed method successfully recovers non-stationary
and heteroscedasticity in the data and provides UQ, which is closer to the one provided by
the treed GP model and at a fraction of the cost.
The results from Tables 1 to 3 are presented in a more digestible format in Figure 4.
Figure 4 (as well as Tables 1 to 3) shows that proposed method provides the best balance
between low computational cost and high accuracy, and good uncertainty quantification
(nominal coverage and tight CIs) in the examples considered.
15
Figure 4: Left: Accuracy vs Time (on log-scale – note purple/circled data). CCR delivers
comparable/higher accuracy, with comparable/smaller cost. Right: Empirical coverage vs Average
length of 95% CIs. CCR provides judicious UQ.
Table 4: Treed GP accuracy (R2
) on the test data, Wall-clock time, Empirical coverage (EC95),
and average length of 95% CIs (CI¯
95).
Metric R
2
, % Time, s EC95, % CI¯
95
Motorcycle 80.14 30.3 97.86 0.83
Nasa 98.39 3987.8 97.16 0.28
Higdon 99.95 329.5 98.81 0.04
Bernholdt 99.68 18363.6 96.42 0.21
kin40k - - - -
χ150k - - - -
5 Conclusion
We have proposed a novel MoE, which combines powerful DNNs to flexibly determine the local regions and sparse GPs to probabilistically model the local regression functions. Through
various experiments, we have demonstrated that this combination provides a flexible, robust
model that is able to recover challenging behaviors such as discontinuities, non-stationarity,
and non-normality and well calibrated uncertainty. In addition, we have established a novel
connection between the maximization-maximization algorithm and the recently introduced
CCR algorithm. This allows us to obtain a fast approximation that significantly outperforms competing methods. Moreover, in some cases, the solution can be further refined
through additional MM iterations. While we focus on the proposed deep mixture of sparse
GP experts, this connection can be generally applied to other MoE architectures for fast
approximation. Future research will explore extensions to infinite mixtures of experts, that
allow for a data-driven number of clusters which can grow unboundedly with the data. In
addition, the recent work of (Rossi et al., 2021) found that combining the FITC approximation of GPs with Bayesian treatment of the inducing points and hyperparameters can
improve performance significantly, in particular, compared with variationally sparse GPs;
in this direction, future work will also explore suitable priors for inducing variables and GP
hyperparameters in MAP estimation.
