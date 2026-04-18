Differentiable Expectation-Maximisation and Applications to
Gaussian Mixture Model Optimal Transport
Samuel Boïté*1
, Eloi Tanguy*1
, Julie Delon1
, Agnès Desolneux2
, and Rémi Flamary3
1Université Paris Cité, CNRS, MAP5, F-75006 Paris, France
2Centre Borelli, CNRS and ENS Paris-Saclay, F-91190 Gif-sur-Yvette, France
3CMAP, CNRS, Ecole Polytechnique, Institut Polytechnique de Paris
September 30, 2025
*: equal contribution.
Abstract
The Expectation-Maximisation (EM) algorithm is a central tool in statistics and machine learning, widely used for latent-variable models such as Gaussian Mixture Models (GMMs). Despite
its ubiquity, EM is typically treated as a non-differentiable black box, preventing its integration
into modern learning pipelines where end-to-end gradient propagation is essential. In this work,
we present and compare several differentiation strategies for EM, from full automatic differentiation to approximate methods, assessing their accuracy and computational efficiency. As a key
application, we leverage this differentiable EM in the computation of the Mixture Wasserstein
distance MW2 between GMMs, allowing MW2 to be used as a differentiable loss in imaging and
machine learning tasks. To complement our practical use of MW2, we contribute a novel stability
result which provides theoretical justification for the use of MW2 with EM, and also introduce a
novel unbalanced variant of MW2. Numerical experiments on barycentre computation, colour and
style transfer, image generation, and texture synthesis illustrate the versatility of the proposed
approach in different settings.
Table of Contents
1 Introduction 2
2 Differentiation of the Expectation-Maximisation Algorithm 3
2.1 Main Ideas of the EM Algorithm . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 3
2.2 Fixed-Point Formulation and Differentiability . . . . . . . . . . . . . . . . . . . . . . . 4
2.3 Gradient Computation Methods . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 5
3 Gaussian Mixture Model Optimal Transport 8
3.1 Reminders on GMM-OT . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 8
3.2 Stability of MW2
2 With Respect to GMM Parameters . . . . . . . . . . . . . . . . . . . 8
3.3 Minimisation of EM − MW2
2
: Local Optima and Weight Fixing . . . . . . . . . . . . . 10
3.4 Unbalanced GMM-OT . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 11
4 Illustrations and Quantitative Study of Gradient Methods 11
4.1 Practical Implementation . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 11
4.2 Flow of EM − MW2
2 with Fixed Weights in 2D . . . . . . . . . . . . . . . . . . . . . . 12
4.3 Flow of EM − MW2
2
in 2D: Discussion on Uniform Weights . . . . . . . . . . . . . . . 12
4.4 Stochastic EM − MW2
2 Flow with Fixed Weights . . . . . . . . . . . . . . . . . . . . . 13
4.5 Quantitative Study of EM Convergence and Gradients . . . . . . . . . . . . . . . . . . 13
1
arXiv:2509.02109v2 [cs.LG] 29 Sep 2025
Differentiable EM and OT S. Boïté, E. Tanguy, J. Delon, A. Desolneux and R. Flamary
5 Applications of Differentiable EM 14
5.1 Barycentre Flow in 2D . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 15
5.2 Colour Transfer . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 15
5.3 Neural Style Transfer . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 16
5.4 Texture Synthesis . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 17
A Supplementary Material 25
A.1 Postponed Proofs . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 25
A.2 Specific GMMs Used in Section 3.2 and Section 4.5 . . . . . . . . . . . . . . . . . . . . 26
A.3 Discussion on Gradient Ground Truths . . . . . . . . . . . . . . . . . . . . . . . . . . . 26
A.4 Explicit Differential Expressions . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 27
A.5 Local Minima in (GMM)-OT . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 32
A.6 Differentiating the Matrix Square Root . . . . . . . . . . . . . . . . . . . . . . . . . . . 38
A.7 Experimental Details and Additional Results . . . . . . . . . . . . . . . . . . . . . . . 39
1 Introduction
The Expectation-Maximisation (EM) algorithm [DLR77] is a ubiquitous tool in statistics to fit mixture models on data [BE94; End03; VH10; Ng13]. Numerous variants of the EM algorithm have been
proposed in the statistics and machine learning communities [DH97; Fri98; FH02; CJJ05; GTG07;
VR08; CM09; Cap11; SCR12; GVS17; ZAC21; Kim22], and are the focus of various monographs
[MK07; MP00]. From a theoretical standpoint, the EM algorithm is only known to converge under
specific conditions [Wu83; Boy83; MK07; XHM16], and its behaviour is still not completely understood in full generality. In machine learning, Gaussian priors have been used for latent space
representations [Ras03; KW14; RMW14; HJA20] with resounding success, while Gaussian Mixture
Models (GMMs) are used more sparingly [NB06; VM19; Yua+20]. Beyond the difficulties of GMM estimation, the core challenge is that the EM algorithm is not easily integrated into end-to-end learning
pipelines, as its differentiation with respect to the input data is not straightforward. Theoretical ties
between the EM algorithm and an alternate minimisation of an energy involving Entropic Optimal
Transport [Cut13] were recently highlighted [RW18; Men+20; Die+24; VL25]. To our knowledge,
these connections do not provide insight into the differentiation of the EM algorithm. In this paper,
we tackle the theory and practice of differentiating the EM algorithm, in the hope of sparking further
research in this direction, beyond the various applications that we present.
One of the core contributions that sparked the Machine Learning wave is automatic differentiation
[Wen64; Lin70; RHW86; GW08], which is a powerful method for complex optimisation problems. In
the setting where the target objective is itself an optimisation problem, the problem is referred to
as “bi-level”. Numerous automatic differentiation methods for bi-level iterative minimisation were
studied in [Gil92; Bec94; Sha+19; MO20; BPV22; BPV24]. Another approach to bi-level optimisation
involving fixed-point problems is the implicit method [Lui+18; BKK19; LVD20; Bol+21; Blo+22;
ER24].
In order to illustrate the potential of differentiable EM, we apply it to imaging tasks which rely
on the comparison of GMMs with Optimal Transport (OT) [Mon81; Kan42]. Specifically, we leverage a variant of the Wasserstein distance between GMMs, called the Mixture-Wasserstein distance
MW2 [DD20], which compares GMMs by matching their components using a small-scale discrete
OT problem that can be solved efficiently [PC19]. The Mixture-Wasserstein distance is one of many
examples of recent advances in computational OT, where less costly surrogates of the Wasserstein
distance such as regularised transport [Cut13] or sliced transport [Bon13] have seen a wide range
of success in machine learning [Bon+11; Cou+17; Kol+19; KLA19; Fey+19]. Computing transport
distances between empirical distributions remains challenging when the number of samples or the
dimensionality of the space becomes too large, even though several solutions have been proposed in
the literature [WB19; Gen+19; Chi+20].
The Mixture-Wasserstein distance has been used in texture synthesis [LDD23], for the evaluation
2
Differentiable EM and OT S. Boïté, E. Tanguy, J. Delon, A. Desolneux and R. Flamary
of generative neural networks [Luz+23], in quantum chemistry [Dal+23], and for domain adaptation [MMS24]. An efficient barycentre computation algorithm for this metric was also recently
proposed in [TDG24]. When using MW2 on discrete data, the space dimension d and the number of
samples n only appear in two stages of the whole computation: the GMM inference on the data, and
the computation of Bures distances between the covariance matrices of the GMM components. This
makes the approach highly versatile and robust to dimensionality in practice. Nevertheless, a current
limitation of the MW2 distance is the inference of the GMMs, which our work renders differentiable
with EM, thus allowing the use of MW2 as a differentiable loss function between datasets in machine
learning tasks.
Objectives. Our goal in this paper is to propose different ways to differentiate EM, allowing for
applications in imaging or machine learning problems. As a focal application, we will pair differentiable EM with the Mixture-Wasserstein distance MW2, which is not difficult to differentiate in
practice (using classical results on the differentiation of discrete OT [PC+19], see also [TDD25,
Proposition 4.3]). Differentiation of the Expectation-Maximisation algorithm is a more involved process. Surprisingly, while EM is well-known and extensively studied, the question of its differentiation
seldom appears in the literature. To the best of our knowledge, the first work in this direction is
[Kim22], which rewrites a Bayesian variant of EM which is related to the Optimal Transport Kernel
Embedding [Mia+21]. In this paper, we propose several approaches for this differentiation, exact via
automatic differentiation1 or approximate, and compare their performance on different applications,
ranging from toy examples to larger-scale machine learning tasks. Given that our contribution is
methodological in nature, our goal is not to achieve state-of-the-art results on these applications, but
rather to illustrate the versatility of the proposed approach.
Paper outline. In Section 2, we begin by recalling the EM algorithm and expressing it as a
fixed point problem. We give precise mathematical meaning to the differentiation of a solution
of the EM algorithm, and present numerous strategies to compute the differential of T steps of
the method with respect to the input data. In Section 3, we provide a short reminder on the
Mixture-Wasserstein distance MW2 and show a stability result for the estimation of MW2 between
GMMs. We discuss practical difficulties in the differentiation of the MW2 distance between GMMs
estimated from data, and provide rationale for the importance of fixing EM weights. To circumvent
the numerical difficulties incurred by weight optimisation and to ensure robustness to Gaussian
outliers, we introduce an unbalanced variant of MW2. In Section 4 we illustrate our methods on the
flow of MW2 composed with the EM algorithm, and perform a quantitative study on the convergence
of the EM algorithm and the quality of the gradient approximations. In Section 5, we present several
applications of differentiable EM: barycentre computation, colour transfer with inspiration from
[Rab+12], style transfer in the spirit of [GEB15], image generation through MW2-based Generative
Adversarial Networks, and a novel texture synthesis method related to [GLR18; LDD23; Hou+23].
2 Differentiation of the Expectation-Maximisation Algorithm
2.1 Main Ideas of the EM Algorithm
The Expectation-Maximisation (EM) algorithm [DLR77] attempts to fit a GMM to a dataset X =
(x1, · · · , xn) ∈ R
n×d
, with a fixed number of components K. We introduce the (hidden) quantities
Y ∈ J1, KK
n which encode the component index of each sample xi
. The GMM parameters θ :=
(w,(mk)k,(Σk)k) lie in the space Θ := △K × (R
d
)
K × (S
++
d
(R))K, where △K is the K-simplex
defined as
△K := (
w ∈ (0, 1)K |
X
K
k=1
wk = 1)
, (1)
1barring numerical approximation, which in certain cases may cause substantial errors.
3
Differentiable EM and OT S. Boïté, E. Tanguy, J. Delon, A. Desolneux and R. Flamary
and S
++
d
(R) is the set of symmetric positive definite matrices. We will denote by µ(θ) the GMM
probability measure of parameters θ. Given X ∈ R
n×d and Y ∈ J1, KK
n
, the complete likelihood and
its logarithm are respectively
Lθ(X, Y ) = Yn
i=1
Y
K
k=1
(wkgmk,Σk
(xi))1(Yi=k)
, ℓθ(X, Y ) = Xn
i=1
X
K
k=1
1(Yi = k) log (wkgmk,Σk
(xi)), (2)
where gmk,Σk
is the Gaussian density with mean mk and covariance Σk (recalled in Eq. (24)). Note
that ℓθ cannot be optimised in θ directly since we do not know the hidden variables Y . The EM
algorithm [DLR77] maximises the log-likelihood by iterating two steps over θt
, first computing the
“responsibilities” γik(θt) which are the posterior probabilities of the hidden quantities Yi given the
data X and the current parameters θt = (w
(t)
,(m
(t)
k
)k,(Σ(t)
k
)k) ∈ Θ:
γik(θt) = P(X,Y)∼µ(θt)⊗n [Yi = k|X = X] = 
w
(t)
k
gm
(t)
k
,Σ
(t)
k
(xi)

/
"X
K
ℓ=1
w
(t)
ℓ
gm
(t)
ℓ
,Σ
(t)
ℓ
(xi)
#
. (3)
The next iteration θt+1 corresponds to a maximisation of ℓθ(X, Y ) with the unknown variables Y
replaced by the posterior probabilities γik(θt), which leads to the following closed-form expressions
for the parameters:
w
(t+1)
k =
1
n
Xn
i=1
γik(θt), m
(t+1)
k =
Xn
i=1
γik(θt)xi
Xn
j=1
γjk(θt)
, Σ
(t+1)
k =
Xn
i=1
γik(θt)(xi − m
(t+1)
k
)(xi − m
(t+1)
k
)
⊤
Xn
j=1
γjk(θt)
.
(4)
It is a standard result (see [Moo96]) that the log-likelihood ℓθ(X) = Pn
i=1 log PK
k=1 wkgmk,Σk
(xi)

increases with respect to θ = (w,(mk),(Σk)) with each EM iteration. In practice, we shall see that
it is sometimes preferable to tweak the standard EM algorithm by not updating the weights w and
keeping the weights w0 of the initialisation θ0 (we refer to this as fixing the weights). We summarise
the two algorithms in Algorithms 1 and 2, with the difference highlighted in red.
Algorithm 1: EM Algorithm
Input: θ0 ∈ Θ, X ∈ R
n×d
, T, K.
1 for t ∈ J0, T − 1K do
2 Expectation: Compute the
responsibilities γ(θt) using Eq. (3);
3 Maximisation: Update
θt+1 = (w
(t+1)
,(m
(t+1)
k
)k,(Σ(t+1)
k
)k)
using Eq. (4);
Algorithm 2: Fixed-Weights EM
Input: θ0 ∈ Θ, X ∈ R
n×d
, T, K.
1 for t ∈ J0, T − 1K do
2 Expectation: Compute the
responsibilities γ(θt) using Eq. (3);
3 Maximisation: Update
θt+1 = (w0,(m
(t+1)
k
)k,(Σ(t+1)
k
)k) using
Eq. (4);
For applications minimising a loss involving the output of the EM algorithm with respect to the data
X, it is important to highlight that even in the fixed-weights version (Algorithm 2), the responsibilities
γ evolve with the EM steps and with the updates of X. As a result, running the EM algorithm along
with X updates is paramount, and it is not sufficient to keep an initial responsibility assignment γ.
2.2 Fixed-Point Formulation and Differentiability
The goal of this section is to express an EM step as a fixed-point operation. For this, a technical
condition is required to ensure that the term Σ
(t+1)
k
is invertible (symmetry and non-negativity of
the eigenvalues is immediate), which requires a slightly stronger condition than assuming that the
points (xi) span R
d
. Since the terms γi,k are positive, we can write the condition as:
X = (x1, · · · , xn) ∈ X := n
(x1, · · · , xn) ∈ (R
d
)
n
| ∀y ∈ R
d
, Span((xi − y)i∈J1,nK
) = R
d
o
. (5)
4
Differentiable EM and OT S. Boïté, E. Tanguy, J. Delon, A. Desolneux and R. Flamary
For X ∈ X and θ0 ∈ Θ, the next iteration θ1 obtained using Eq. (4) satisfies θ1 ∈ Θ. Note that if
ν is an absolutely continuous probability measure on R
d
, and if n ≥ d + 1, then Eq. (5) is verified
almost surely for X ∼ ν
⊗n
. This condition can be seen as a weaker variant of being in a “standard
configuration”. It can be shown that X is an open subset of (R
d
)
n ≃ R
n×d
.
We study the map F : Θ × X −→ Θ that maps a parameter θ to the value of the next M-step using
Eq. (4). For convenience, we also write FX := F(·, X) and F
t
X the t-th iteration FX ◦ · · · ◦ FX for
t ∈ N. An optimal solution to the EM algorithm can be seen as a fixed point of F, i.e. θ
∗ = FX(θ
∗
).
Numerically, one takes a final iterate θT of the iteration scheme
∀t ∈ J0, T − 1K, θt+1 = F(θt
, X),
with an arbitrary initialisation θ0 ∈ Θ. Due to the definition of Θ as the set of parameters with
weights wk ∈ (0, 1) and positive definite matrices Σk, the map F is of class C∞ (jointly in (θ, X)), by
virtue of the explicit expressions Eq. (4).
We now aim to give meaning to the gradient with respect to the data of a “true” solution of the
EM algorithm. To this end, we need to work under the assumption of convergence of EM to a
non-degenerate fixed point:
Assumption 1. There exists (θ0, X0) ∈ Θ × X such that θ
∗
(θ0, X0) := lim t−→+∞
F
t
X0
(θ0) exists in
Θ, with ∂F
∂θ (θ
∗
(θ0, X0), X0) − I invertible.
While convergence is typically observed in practice numerically, from a theoretical standpoint, it is
a delicate matter. See [MK07, Chapter 3] for a reference on this field of research. We are now ready
to formulate Proposition 2.1, which shows that the gradient of an EM solution with respect to the
data is well-defined, using the implicit function theorem.
Proposition 2.1. Under Assumption 1, there exist open neighbourhoods Θ∗
, X0 such that
θ
∗
(θ0, X0) ∈ Θ∗ ⊂ Θ and X0 ∈ X0 ⊂ X , where there exists θ
∗
(θ0, ·) ∈ C∞(X0, Θ∗
) with:
∀(θ, X) ∈ Θ
∗ × X0, F(θ, X) = θ ⇐⇒ θ = θ
∗
(θ0, X). (6)
Proof. Let G := (
Θ × X −→ Θ
(θ, X) 7−→ F(θ, X) − θ
. Thanks to the regularity of F, G is of class C∞.
Assumption 1 implies that G(θ
∗
(X0), X0) = 0, with ∂G
∂θ (θ
∗
(X0), X0) invertible.
By the implicit function theorem, there exist open neighbourhoods Θ∗
, X0 such that θ
∗
(θ0, X0) ∈
Θ∗ ⊂ Θ and X0 ∈ X0 ⊂ X , and there exists g : X0 −→ Θ∗ of class C∞ such that g(X0) = θ
∗
(θ0, X0),
and:
∀(θ, X) ∈ Θ
∗ × X0, F(θ, X) = θ ⇐⇒ θ = g(X).
For X ∈ X0, we define θ
∗
(θ0, X) := g(X).
To alleviate notation, we will write simply θ
∗
(X) := θ
∗
(θ0, X) and continue with Assumption 1
and the map θ
∗
from Proposition 2.1. For the sake of completeness and to pave the way for future
theoretical study, we provide the explicit expressions of the partial differentials of F in Appendix A.4.
Note that from a statistical viewpoint, the parameter θ
∗
is not a Maximum-Likelihood Estimator
(which, in fact, does not exist for GMMs), it is only the output of the EM algorithm which is often
a local maximum of the likelihood.
2.3 Gradient Computation Methods
In this section, we are concerned with practical implementation of the computation of the gradient of
the EM algorithm with respect to the data ∂X[F
T
X (θ0)], given some initialisation θ0 ∈ Θ and number
5
Differentiable EM and OT S. Boïté, E. Tanguy, J. Delon, A. Desolneux and R. Flamary
of iterations T ≥ 1. In addition to the automatic differentiation method, we present two alternative
approximate strategies which rely only on the last parameter θT . These methods work under the
assumption that θT (X) ≈ θ
∗
(X), where θ
∗
is defined in Proposition 2.1 and refers to the fixed point
limt−→+∞ F
t
X(θ0), assuming convergence.
Full Automatic Differentiation (AD). The most naïve approach consists in computing the
gradient through all iterations using the backpropagation algorithm (for instance, using PyTorch’s
automatic differentiation [Pas+19]). In other words, the “full automatic gradient” corresponds to
letting automatic differentiation compute ∂X[F
T
X (θ0)] directly, using an automatically differentiable
implementation of the EM algorithm. For a large number of iterations T, AD can be considered
as a natural baseline for computing the exact gradient, up to numerical precision. It is still an
approximation, due to propagation of numerical errors, but is to the best of our knowledge the best
available approximation of the true gradient, as discussed in Appendix A.3. Nevertheless, we use
AD as a natural baseline method for comparisons. The AD method may be costly (both in time and
memory) if the number of iterations T is very large.
Approximate Implicit Gradient (AI). Our goal is to approximate the gradient ∂θ∗
∂X (X) at a
fixed point θ
∗ of F(·, X). Thanks to the differentiability property of Proposition 2.1 and under
Assumption 1, we can differentiate with respect to X using the chain rule:
∂θ∗
∂X(X) =
∂
∂X [F(θ
∗
, X)] =
∂F
∂θ (θ
∗
, X)
∂θ∗
∂X(X) +
∂F
∂X(θ
∗
, X). (7)
We deduce the following equation on
∂θ∗
∂X(X):

I −
∂F
∂θ (θ
∗
, X)
!
∂θ∗
∂X(X) =
∂F
∂X(θ
∗
, X), (8)
using Assumption 1 again, we can invert the matrix on the left hand-side, yielding
∂θ∗
∂X(X) =
I −
∂F
∂θ (θ
∗
, X)
!−1
∂F
∂X(θ
∗
, X). (9)
We define the approximate implicit gradient by approximating θ
∗ ≈ θT :
JAI :=
I −
∂F
∂θ (θT , X)
!−1
∂F
∂X(θT , X) (10)
The implicit approximation is theoretically exact (barring numerical imprecision in the inversion in
particular) when θT = θ
∗
, however it requires additional costly computations: first evaluate the differential ∂θF(θT , X), and then solve a large linear system to compute (I −∂θF(θ
∗
, X))−1∂XF(θ
∗
, X).
One-Step Gradient Approximation (OS). The One-Step method (OS) studied in [BPV24]
(within a particular framework of bi-level optimisation), works under the following condition:
Condition 1.






∂F
∂θ (θ, X)






op
≤ ρ ≪ 1 for any θ, X.
In the case of the EM algorithm, Condition 1 is not verified, since the eigenvalues of the partial
differential ∂F
∂θ (θ, X) are commonly larger than 1, even in the neighbourhood of a fixed point. Under
Condition 1, the OS approximation further neglects the term (I −∂θF(θ
∗
, X))−1
in Eq. (10), yielding
the following expression:
JOS :=
∂F
∂X(θT −1, X) ≈
∂F
∂X(θ
∗
, X) ≈

I −
∂F
∂θ (θ
∗
, X)
!−1
∂F
∂X(θ
∗
, X) =
∂θ∗
∂X(X). (11)
6
Differentiable EM and OT S. Boïté, E. Tanguy, J. Delon, A. Desolneux and R. Flamary
In practice, the OS method corresponds to computing the gradient of the EM output θT (X) =
F
T
X (θ0) with respect to the data X only through the last iteration θT (X) = FX(θT −1), neglecting the
dependence of the penultimate iteration θT −1 on X. Using automatic differentiation (for example
with PyTorch [Pas+19]), this is done conveniently by computing θT −1 = F
T −1
X (θ0) without gradient
computation (e.g. with torch.no_grad()), and performing the last step with gradient computation.
Due to this computation method, the OS gradient is numerically inexpensive compared to the others,
albeit at the expense of precision. See [BPV24] for a detailed presentation.
Method Time Memory
Full Automatic Differentiation (AD) O

T(nKd2 + Kd3
)

O(TKd2 + nd)
Approximate Implicit Gradient (AI) O

T nKd2 + K3d
6 + nK2d
5

O(nK2d
4
)
One-Step gradient (OS) O

T(nKd2 + Kd3
)

O(Kd2 + nd)
Table 1: Complexities of the backward passes (gradient computations) for T EM iterations on n
points in R
d with K components.
Time complexity and memory footprint. The complexities of the gradient computation approaches are summarised in Table 1. As a baseline, the time complexity of the forward pass (EM
algorithm without gradients) is O

T(nKd2 +Kd3
)

, while its memory footprint is O(Kd2 +nd). The
complexities are deduced from the differential expressions in Appendix A.4. The O(Kd3
) factor corresponds to inverting the K covariance matrices of size d×d during each E-step. The O(nKd2
) factor
comes from the M-step parameter updates (weights, means, covariances) and the differentiation of
these updates with respect to X or θ.
The Warm-Start Method for Iteration of Differentiable EM. In many practical applications, we are interested in minimising a certain loss function L applied to the output θT (X) of the EM
algorithm2
. In this case, after a (small) gradient descent step Xt+1 computed from Xt
, the output
of the EM algorithm with data Xt will often be a good initialisation for the EM algorithm with data
Xt+1. As a result, we suggest operating the EM algorithm with only one iteration and using the
output of the previous step as an initialisation. This leads to the following algorithm, which we refer
to as the Warm-Start EM Flow of a loss L : Θ −→ R.
Algorithm 3: Warm-Start EM Flow
Input: θ0 ∈ Θ, X0 ∈ X , TGD ∈ N, L : Θ −→ R.
1 for t ∈ J0, TGD − 1K do
2 θt+1 = F(θt
, Xt);
3 Xt+1 = Xt − ηt
∂L
∂θ (θt)
∂F
∂X (θt
, Xt);
The gradient step at line 3 means that we perform automatic differentiation of the expression
L(F(θt
, Xt)) with respect to X at Xt
, seeing θt as a constant and storing the value θt+1 = F(θt
, Xt) for
the next iteration. In essence, the Warm-Start method corresponds to an online OS gradient, which
does not suffer from the approximation error of the OS method (since only one step is performed),
yet benefits from the low memory footprint of the OS method.
2
for example L(θT (X)) = MW2
2(µ(F(θT , X)), ν), where ν is a target GMM, see Section 3.
7
Differentiable EM and OT S. Boïté, E. Tanguy, J. Delon, A. Desolneux and R. Flamary
3 Gaussian Mixture Model Optimal Transport
3.1 Reminders on GMM-OT
This section summarises the main results from [DD20]. The quadratic Wasserstein distance between
two probability measures µ0 and µ1 on R
d with finite second moments is defined by
W2
2
(µ0, µ1) := inf
π∈Π(µ0,µ1)
ˆ
Rd×Rd
∥y0 − y1∥
2
2dπ(y0, y1), (12)
where Π(µ0, µ1) denotes the set of probability measures with finite second moments on R
d×R
d whose
marginals are µ0 and µ1. A solution π
∗
to Eq. (12) is called an optimal transport plan between µ0
and µ1. This distance has been widely used over the past fifteen years for various applications in data
science. Let GMMd denote the set of probability measures that can be written as finite Gaussian
Mixture Models (GMMs) on R
d
. Transport plans and barycentres between GMMs with respect
to W2 are generally not GMMs, which is a limitation when such representations are used for data
analysis or generation. For this reason, the authors of [DD20] propose to modify the W2 formulation
by restricting the couplings to be GMMs on R
d × R
d
. More precisely, given µ0, µ1 ∈ GMMd, one can
define
MW2
2
(µ0, µ1) := inf
π∈ΠGMM(µ0,µ1)
ˆ
R2d
∥y0 − y1∥
2
2dπ(y0, y1), (13)
where ΠGMM(µ0, µ1) denotes the set of probability measures in GMM2d with marginals µ0 and µ1.
The problem is well-defined since this set contains the product measure µ0 ⊗ µ1. The authors show
that MW2 defines a distance between elements of GMMd. Moreover, if µ0 =
PK0
k=1 w
(0)
k
µ
(0)
k
and
µ1 =
PK1
ℓ=1 w
(1)
ℓ
µ
(1)
ℓ
, where (w
(0)
k
)k ∈ △K0
and (w
(1)
ℓ
)ℓ ∈ △K1
, and µ
(0)
k
, µ
(1)
ℓ
are Gaussian measures,
then it can be shown ([DD20, Proposition 4]) that
MW2
2
(µ0, µ1) = min
P ∈Π(w0,w1)
X
k,ℓ
PklW2
2
(µ
(0)
k
, µ
(1)
ℓ
), (14)
where Π(w0, w1) is the set of K0 × K1 matrices with non-negative entries and marginals w0 and w1:
Π(w0, w1) =



P ∈ MK0,K1
(R
+); ∀k,X
j
Pkj = w
(0)
k
and ∀j, X
k
Pkj = w
(1)
j



.
This discrete formulation makes MW2 very easy to compute in practice, even in high dimensions.
Indeed, the W2 distance between two Gaussian measures µ = N (m, Σ) and µ˜ = N ( ˜m, Σ) ˜ admits a
closed-form expression:
W2
2
(µ, µ˜) = ∥m − m˜ ∥
2
2 + tr 
Σ + Σ˜ − 2

Σ
1
2 ΣΣ˜
1
2
 1
2

, (15)
where M
1
2 denotes the unique positive semidefinite square root of the positive semidefinite matrix
M. If the parameters of the GMMs µ0 and µ1 are known, computing Eq. (14) amounts to evaluating
K0 × K1 Wasserstein distances between Gaussians and solving a discrete transport problem of size
K0 × K1. It is also possible to define barycentres for MW2 [DD20; TDG24], which leads to a similar
discrete formulation. Given point clouds, [DD20] suggest using EM to fit GMMs to the data, allowing
comparison of the point clouds with EM-MW2
2
.
3.2 Stability of MW2
2 With Respect to GMM Parameters
To study the stability of the MW2 distance with respect to the GMM parameters, we leverage a
discrete OT stability result from [TFD24]. To relate this problem to discrete OT stability, we see
MW2
2
(µ0, µ1) as a particular discrete Kantorovich problem with cost matrix Ck0,k1
:= ∥m
(0)
k0
−m
(1)
k1
∥
2
2+
d
2
BW(Σ(0)
k0
, Σ
(1)
k1
), where we recall the expression of the Bures-Wasserstein distance on S
+
d
(R):
∀Σ, Σ
′ ∈ S
+
d
(R), dBW(Σ, Σ
′
) := q
Tr(Σ + Σ′ − 2(Σ1/2Σ′Σ1/2)
1/2). (16)
8
Differentiable EM and OT S. Boïté, E. Tanguy, J. Delon, A. Desolneux and R. Flamary
We show that if the GMM parameters are sufficiently close (thanks to EM convergence for instance),
then the MW2
2
costs will also be close thanks to the stability result from [TFD24]. While general sample complexity results for the EM algorithm are not available, assuming a certain rate of convergence
towards the true parameters, this result shows that the precision obtained using EM translates into a
precision on the MW2
2 distance. This key observation is a first step towards guaranteeing the quality
of MW2
2 as a loss function with respect to the data. We formulate two results: first, Proposition 3.1
quantifies the decrease of MW2
2
(ˆµ, µ) when µˆ is an estimator of µ; then, Proposition 3.2 quantifies
the decrease of |MW2
2
(ˆµ0, µˆ1) − MW2
2
(µ0, µ1)| when µˆi are estimators of µi for i ∈ {0, 1}. We defer
the proofs to Appendix A.1.
Proposition 3.1. Consider GMM parameters ( ˆw, m, ˆ Σ) ˆ ∈ △K × R
K×d × S
++
d
(R)
K of a GMM
µˆ as estimators of (w, m, Σ) ∈ △K × R
K×d × S
++
d
(R)
K which are parameters of a target GMM
µ. Assume “convergence rates” on the parameter estimations for k ∈ J1, KK:
E [∥w − wˆ∥1] ≤ ρw, E
h
W2
2
(N ( ˆmk, Σˆ
k), N (mk, Σk))i
≤ ρN ,
then the following stability bound holds:
E
h
MW2
2
(ˆµ, µ)
i
≤ ρN +
ρw
2
max
k,ℓ
E
h
W2
2
(N ( ˆmk, Σˆ
k), N (mℓ
, Σℓ))i
. (17)
Proposition 3.2. For i ∈ {0, 1}, consider GMM parameters ( ˆwi
, mˆ i
, Σˆ
i) ∈ △Ki × R
Ki×d ×
S
++
d
(R)
Ki of a GMM µˆi as estimators of (wi
, mi
, Σi) ∈ △Ki × R
Ki×d × S
++
d
(R)
Ki which are
parameters of a target GMM µi
. Assume that the means and covariances are bounded, namely
that there exists Rm > 0, RΣ > 0 such that:
∀i ∈ {0, 1}, ∀k ∈ J1, KiK, ∥m
(i)
k
∥2 ≤ Rm, ∥mˆ
(i)
k
∥2 ≤ Rm,
q
TrΣ(i)
k ≤ RΣ,
q
TrΣˆ
(i)
k ≤ RΣ.
Further assume “convergence rates” on the parameter estimations k ∈ J1, KiK:
∀i ∈ {0, 1}, E [∥wi − wˆi∥1] ≤ ρw, E
h
∥m
(i)
k − mˆ
(i)
k
∥2
i
≤ ρm, E
h
dBW 
Σ
(i)
k
, Σˆ
(i)
k
i ≤ ρΣ,
then the following stability bound holds:
E
h

MW2
2
(ˆµ0, µˆ1) − MW2
2
(µ0, µ1)



i
≤ 8Rmρm + 8RΣρΣ + 8(R
2
m + R
2
Σ)ρw. (18)
We complement the stability bounds of Proposition 3.1 and Proposition 3.2 with an empirical study.
We fix K = 3, d = 2, vary the sample size n between 103 and 2 · 104
, and for each n run 40
repetitions of EM with 200 iterations (with k-means++ initialisation), noting the resulting estimates
µˆn and νˆn. Three regimes of component separation (low, medium, high) are considered, controlled
by a scale parameter σ applied to the covariances. The mixtures µ, ν are fixed, and we provide a
visualisation in Appendix A.2. For each n, we report the median error and interquartile range across
repetitions. Fig. 1a shows MW2
2
(ˆµn, µ). With medium and high component separation, the error
decreases regularly with n, while with low separation it remains flat, indicating that EM does not
improve the estimation significantly in this regime. Fig. 1b shows the relative error |MW2
2
(ˆµn, νˆn) −
MW2
2
(µ, ν)|/MW2
2
(µ, ν). Medium and high separation again yield decay with n, while low separation
plateaus. The results above translate parameter error rates of EM into rates for MW2
2
, but do not
specify a universal value. In well-separated special cases, EM is known to achieve a rate of O(n
−1/2
)
(see [KC20] for spherical covariances). When separation is weak, EM appears not to converge reliably,
hence the MW2
2
error does not decrease.
9
Differentiable EM and OT S. Boïté, E. Tanguy, J. Delon, A. Desolneux and R. Flamary
10
3 10
4
sample size n
10
−2
10
−1
10
0
10
1
M
W2
2
( ̂ μn, μ)
low
medium
high
n
−1/2
(a) One-sample: MW2
2
(ˆµn, µ)
10
3 10
4
sample size n
10
−2
10
−1
10
0
|M
W2
2
( ̂ μn, ̂ νn) −
M
W2
2
(μ, ν)|/M
W2
2
(μ, ν)
low
medium
high
n
−1/2
(b) Two-sample: relative error
Figure 1: Sample complexity of MW2
2
for three specific GMMs of “low” to “high” mode separation.
Curves show the median across 40 repetitions, with shaded interquartile ranges.
3.3 Minimisation of EM − MW2
2
: Local Optima and Weight Fixing
In practice, the minimisation of the energy X 7−→ MW2
2
(F
T
X (θ0), ν) for some initialisation θ0 ∈ Θ and
a target GMM ν comes with numerous challenges. The first hurdle is the “outer” minimisation of the
MW2
2
cost. To illustrate this difficulty, we begin with a study of the simpler energy µ 7−→ W2
2
(µ, ν) for
a fixed (discrete) measure ν ∈ P2(R
d
) with respect to the weights and support of the discrete measure
µ =
Pn
i=1 aiδxi
. This setting corresponds to the optimisation of MW2
2 with known covariances, and
thus highlights practical bottlenecks for the minimisation of the complete energy at stake, EM−MW2
2
.
The objective of this section is to provide a theoretical rationale for fixing the mixture weights in
practical applications, which is to say using Algorithm 2 instead of standard EM (Algorithm 1).
Local Minima of the Discrete 2-Wasserstein Distance. We focus on a particular instance
of the minimisation of µ 7−→ W2
2
(µ, ν), and show the existence of a strict local minimum. We
parametrise a discrete measure µ ∈ P(R) with a support size of 3 as follows:
∀α ∈ [−
1
6
,
1
6
]
2
, ∀η ∈ (−
1
2
,
1
2
)
3
, µα,η := ( 1
6 + α1)δη1 + ( 1
6 + α2)δη2 + ( 2
3 − α1 − α2)δ1+η3
,
and we fix a target measure ν := 1
3
(δ0 + δ1−ε + δ1+ε) for a fixed ε ∈ (0,
1
2
). The energy to minimise
is then:
E3 := (
[−
1
6
,
1
6
]
2 × (−
1
2
,
1
2
)
3 −→ R
(α, η) 7−→ W2
2
(µα,η, ν)
. (19)
Obviously, the energy (α, η) 7−→ W2
2
(µα,η, ν) has a global minimum with value 0 at all (α, η) such
that µα,η = ν. However, on the region with (α, η) ∈ [−
1
6
,
1
6
]
2 × (−
1
2
,
1
2
)
3
, we show in Appendix A.5.1
that the energy E3 has a minimum at α = 0R2 and η = 0R3 , with value E3(0R2 , 0R3 ) > 0. Note that
for the case n = 2 we can show that there is a unique local minimum, see Appendix A.5.2.
Essential Stationary Points for the EM−MW2
2 Loss. We have seen in the previous paragraph
that optimising µ 7−→ W2
2
(µ, ν) with respect to the weights and support of µ can lead to local minima,
which are not global minima. An additional difficulty arises when optimising the energy
EEM−MW2
2
:= X 7−→ MW2
2
(µ(F(θ, X)), ν), (20)
with one iteration F of the EM algorithm, due to the update on the weights. The issue is that
at some problematic points X to which the algorithm often converges in practice, the gradient
∂XEEM−MW2
2
(X) becomes extremely small, in particular when the covariances are highly localised.
This leads in practice to undesirable convergence to an essential local minimum, as illustrated by an
example in Section 4.3. We provide a theoretical explanation in a simple case in Appendix A.5.3. To
10
Differentiable EM and OT S. Boïté, E. Tanguy, J. Delon, A. Desolneux and R. Flamary
avoid these numerical issues, we propose fixing the weights of the GMMs in the EM steps by using
Algorithm 2.
Our theoretical observations suggest that considering GMMs with uniform weights and using fixedweights EM (Algorithm 2) is a more stable alternative to standard EM (Algorithm 1). In practice, we
believe it is also preferable to keep the same number of components K between the compared GMMs
for additional stability. Note that an identifiability issue remains with GMMs: if the means and
covariances of two modes coincide, then the GMM can also be written by fusing both components
and adding their weights. At this stage, it remains unclear whether this phenomenon has an impact
on the optimisation behaviour (note that we never observed it in our experiments).
3.4 Unbalanced GMM-OT
Starting from the discrete formulation of the MW2 distance, we relax the constraints on the transport plan π, penalising the marginal conditions instead of enforcing them in the optimisation problem. The resulting optimisation problem defines an unbalanced GMM-OT distance on the set
GMM+
d
(∞) of GMMs with positive weights on R
d
, as in [LMS18]. Given two Gaussian mixtures
µ =
PK0
k0=1 w
(0)
k0
gk0
, ν =
PK1
k1=1 w
(1)
k1
gk1 ∈ GMM+
d
(∞), and regularisation parameters (λ0, λ1) ∈
(0, +∞)
2
, the unbalanced GMM-OT cost is defined as:
UMW2
2
(µ, ν; λ0, λ1) := min
π∈R
K0×K1
+
X
k0,k1
πk0,k1W2
2
(gk0
, gk1
) + λ0 KL(π1|w0) + λ1 KL(π
⊤1|w1), (21)
where we recall that for a, b ∈ (0, +∞)
K, the Kullback-Leibler divergence is KL(a|b) := P
k ak log( ak
bk
).
We have seen that MW2 is a particular discrete Kantorovich problem, and likewise the unbalanced
GMM-OT distance UMW2 is simply an unbalanced discrete OT problem with a particular cost
matrix.
Given the numerical challenges of optimising the weights in the balanced formulation (see the discussion in Section 3.3), we introduce this variant as a possibly more stable alternative. We suspect
that the underlying geometry on the weights induced by unbalanced OT [LMS18; Chi+18b] is more
amenable to optimisation. Furthermore, unbalanced OT has been shown by [Fat+21a] to be stable
with respect to minibatch sampling, which is paramount for large-scale machine learning applications.
4 Illustrations and Quantitative Study of Gradient Methods
4.1 Practical Implementation
For practical implementation of the EM algorithm, some specific implementation strategies are required to ensure numerical stability, in particular when computing gradients. The first technique is
applied in the E step, and consists in computing the responsibilities γik(θt) in logarithmic space and
using the so-called “log-sum-exp trick”3
to compute the normalisation in Eq. (3). Furthermore, to
stabilise the (differentiable) expression of the Gaussian density gm,Σ(x), we leverage the Cholesky
decomposition of the covariance matrix Σ, which uniquely decomposes Σ = LL⊤ where L ∈ R
d×d
is a lower triangular matrix. In particular, the computation of the inverse is simplified by solving
triangular systems, and the determinant of Σ is simply det Σ = (Q
a Laa)
2
(which we compute in
logarithmic space as well).
Another important implementation aspect concerns a differentiable implementation of the matrix
square root of symmetric positive semidefinite matrices. This is required in the computation of the
Bures distance Eq. (16) for the MW2 distance. Unfortunately, the naive implementation using the
spectral decomposition suffers from numerical instability when eigenvalues are very close in value4
.
3
for instance, see this blog post for an explanation of this well-known trick.
4
as explained in the PyTorch documentation for torch.linalg.eigh().
11
Differentiable EM and OT S. Boïté, E. Tanguy, J. Delon, A. Desolneux and R. Flamary
Leveraging an explicit formula for the gradient of the matrix square root (detailed in Appendix A.6),
we circumvent these numerical issues by implementing our own differentiable square root function
with an explicit gradient.
As is done in scikit-learn’s implementation of the EM algorithm, we have an optional regularisation
term εr ≥ 0 for the covariance matrices Σk to ensure positive-definiteness. The idea is to replace
the update Σ
(t+1)
k with Σ
(t+1)
k + εrId to enforce a minimum eigenvalue of εr. This regularisation was
particularly crucial for numerical stability in higher-dimensional cases where the covariances were
almost singular, which led to exploding gradients. In the larger-scale examples from Section 5, we
chose a heuristic which sets εr := 10−4 ×smax, where smax is the largest eigenvalue of the covariances
of a GMM fitted on the target data.
4.2 Flow of EM − MW2
2 with Fixed Weights in 2D
In this section, we illustrate the use of differentiable EM for OT by numerically computing the flow
(i.e. gradient descent) of the following energy:
ET := X ∈ R
n×2
7−→ MW2
2
(µ(F
T
X (θ0)), ν), (22)
for a fixed target GMM ν, an initialisation θ0 and a number of EM steps T. We use a variant of EM
presented in Algorithm 2 that fixes the mixture weights in this experiment. We will compare three
gradient computation methods to compute (or approximate) the gradient of F
T
X (θ0) with respect to
X, within the gradient descent of ET , performed using automatic differentiation. The setup is as
follows: the initial dataset X ∈ R
200×2
corresponds to samples of a GMM µ0 with 3 components, and
we want to displace this point cloud to match a target GMM ν with 3 components. We represent
the setup in Fig. 2a and the flow for AD method in Fig. 2b.
The results for the AI method are both visually and quantitatively very close, however the experiment
took six times longer to run for AI. We observe satisfactory convergence of the flow of ET towards
the target GMM ν. In many applications involving fixed EM weights (Algorithm 2), we observe that
particles follow rectilinear trajectories, which is a similar behaviour to Wasserstein flows of W2
2
(see
[CNR25, Section 5.3]). We interpret this phenomenon as a consequence of the fixed weights, which
translate to a Lagrangian viewpoint on the GMMs. In simple cases, the MW2
2
-optimal plans between
the GMMs may not change during the flow, and thus the particles are moved along the induced
(rectilinear) trajectories between each GMM component (see [DD20, Proposition 4]). In Fig. 2c, we
show the flow with the OS method, which converges more slowly and to an unsatisfactory stationary
point. This is due to the fact that OS requires a contraction assumption that is not verified for
EM. OS was comparable in computation time to AD. We also experimented with the Warm-Start
flow from Algorithm 3, which is a different minimisation method to minimise ET , yet yielded almost
identical results to the AD, with a 40% lower computation time.
4.3 Flow of EM − MW2
2
in 2D: Discussion on Uniform Weights
We now consider a similar setting to Section 4.2, but without fixing the weights in the EM algorithm,
i.e. using standard EM Algorithm 1. We compare two settings: the first with non-uniform weights
w0 := ( 1
5
,
1
5
,
3
5
) for the initial GMM, and weights w1 := ( 1
2
,
3
10 ,
1
5
) for the target GMM; and the second
with uniform weights for both. In Fig. 3, we show the flow of ET with the AD method. We observe
in Fig. 3a that the flow for non-uniform weights converges to an unsatisfactory local minimum, with
a final GMM weight of [0.41039274, 0.2030173, 0.38658995] instead of the target [0.5, 0.3,
0.2], as shown on the simplex in Fig. 3b. In contrast, the flow for uniform weights presented in
Fig. 3c converges to the target GMM and achieves a substantially lower energy, as reported in Fig. 3d.
The weights stay close to uniform, with a final GMM weight of [0.33314218, 0.30985782, 0.357].
The optimisation failure in the non-uniform case is due to the essential stationary point problem
illustrated in Section 3.3: intuitively, to change the weights of the current GMM, the particles need
12
Differentiable EM and OT S. Boïté, E. Tanguy, J. Delon, A. Desolneux and R. Flamary
source GMM
target GMM
data
data GMM
(a) ET flow setup. (b) AD method. (c) OS method.
Figure 2: Comparison of experimental setup and flows of ET using different methods. The dark
shades of purple correspond to earlier iterations, and the yellow shades to later iterations. (WarmStart and AI are almost identical to AD)
to change components, but this is not possible if the components are too distant. While our framework encompasses the case of non-uniform weights, as illustrated theoretically and experimentally, it
appears that the non-uniform weight setting is impractical. As a result, we recommend using uniform
weights, in particular using the fixed-weights EM approach (Algorithm 2) for speed and stability.
(a) Particle flow (NU).
(1, 0, 0) (0, 1, 0)
(0, 0, 1)
EM Flow Weights with Non-Uniform Weights
source GMM
target GMM
(b) Weight evolution on
the simplex △3 (NU).
(c) Particle flow (U).
0 100 200 300 400 500
Iteration
10
1
10
2
MW2
2 Loss per Iteration
Non-Uniform Weights
Uniform Weights
(d) Energy evolutions.
Figure 3: Flow of ET with the AD method and standard EM Algorithm 1. We compare two settings:
one with non-uniform GMM weights (NU) and one with uniform weights (U).
4.4 Stochastic EM − MW2
2 Flow with Fixed Weights
We consider a similar setting to Section 4.2 but introduce stochasticity in the flow at each step by
performing EM only on a subsample of the optimised source point cloud and of the target point
cloud. While we illustrate the technique on a toy example here, this “minibatch” stochastic gradient
descent method is useful in practice when the dataset size is too large for simultaneous optimisation
[Fat+20; Fat+21b; Fat+21a; Ton+24]. The same principle is applied to the image generation task in
Appendix A.7.6. We observe in Fig. 4 that the general trajectory remains similar to the deterministic
case. Notice that in this setting, the components are sufficiently close together to interact, yielding
non-rectilinear trajectories when points are influenced by multiple components. This is amplified by
the stochasticity of the method.
4.5 Quantitative Study of EM Convergence and Gradients
We study the impact of the number of points n, components K and EM iterations T on the convergence of EM iterations (to a fixed point of F), the local contractivity of F around the fixed point,
and the gradient approximation methods introduced in Section 2.3. The experimental setting is as
13
Differentiable EM and OT S. Boïté, E. Tanguy, J. Delon, A. Desolneux and R. Flamary
Figure 4: Stochastic Flow of ET for Algorithm 2 with the full automatic differentiation method. We
vary the sub-sampling ratio r ∈ (0, 1], which corresponds to performing EM on only [r × n] random
points from the current point cloud at each step.
follows: for each of the three parameters n, K, T separately (say n), we consider a range of values
(say n ∈ {100, 200, 500, 1000, 1500, 2000}) with the others fixed. For each value of the parameter in
this range, and for three different GMMs in GMMd(K) with d = 3 (shown in Appendix A.2), we
sample the data from X ∼ µ
⊗n
0
60 times, then run the EM algorithm for T iterations. We run the
experiments on three different GMMs taken with random parameters (adding a small term 10−14Id
to the covariances to avoid vanishing eigenvalues). Note that GMM# 1 is better conditioned than
GMM# 2, and GMM# 3 is the worst-conditioned. We observe the mean squared error of the fixed
point property F(θT , X) ≈ θT by evaluating 1
p
∥θT −F(θT , X)∥
2
2 with p := K + Kd+ Kd2
, measuring
the quality of convergence of the EM algorithm. To study the local contractivity of F, we compute
the spectral norm ∥∂θF(θT , X)∥op: if this is close to 0, then locally the iterated function FX has a
tame behaviour and the OS method is expected to work well, while if it is close to or larger than 1,
the local landscape is difficult and the OS method is expected to fail. Finally, we compare the OS
and AI gradients (from Eqs. (10) and (11)) to the reference AD gradient by computing the relative
MSEs 1
p
∥JOS − JAD∥
2
2
/(
1
p
∥JAD∥
2
2
) and 1
p
∥JAI − JAD∥
2
2
/(
1
p
∥JAD∥
2
2
), where JAD is the AD gradient,
which serves as a baseline (see Appendix A.3 for a discussion on this choice). Concerning the impact
of the number of components K, we defer to Appendix A.7.1, since the findings are less conclusive.
Impact of the number of samples n. We begin by fixing d = 3, K = 3 and T = 30 and varying
n ∈ {100, 200, 500, 1000, 1500, 2000}. The results are shown in row 1 of Fig. 5, and we observe that
EM appears to converge to a fixed point for all n, albeit with a large variance in the MSE depending
on the sampled GMMs. The spectral norm of the Jacobian is often close to 0.6 and has no clear trend
with n, hence we expect the OS method to be a very coarse approximation of the true gradient. The
quality of the OS gradient is relatively poor, and substantially worse than the AI gradient, whose
median MSE is much smaller, but suffers from very high variance (in log space). Comparing GMMs
shows that a precise EM convergence leads to high precision for the AI gradient.
Impact of the number of EM iterations T. Finally, we fix n = 200, d = 3 and K = 3, and vary
T ∈ {1, 2, 5, 10, 15, 20, 30, 40} in row 3 of Fig. 5. Reassuringly, increasing the number of iterations T
leads to improved convergence of the EM algorithm to better-conditioned points. The convergence
speed seems heavily dependent on the GMM, with an additional variance caused by the dataset
sampling. In the favourable settings for larger T, the AI approximation substantially outperforms
the OS approximation, but suffers from higher variance. Since the spectral norm of the Jacobian
stabilises to values of the order of 0.5, the OS method plateaus at coarse MSEs, even for larger T.
5 Applications of Differentiable EM
In this section, we propose numerous larger-scale applications of the differentiation of the EM algorithm presented in Section 2. Our goal is to illustrate the versatility of the proposed approach, rather
than to achieve state-of-the-art results on these applications. In Appendix A.7.6, we also present an
14
Differentiable EM and OT S. Boïté, E. Tanguy, J. Delon, A. Desolneux and R. Flamary
500 1000 1500 2000
n
10
26
10
22
10
18
10
14
10
10
10
6
Fixed Point MSE
500 1000 1500 2000
n
10
1
Max Eigenvalue of F/
500 1000 1500 2000
n
10
4
10
3
10
2
10
1
10
0
relative MSE: OS vs AD
500 1000 1500 2000
n
10
28
10
23
10
18
10
13
10
8
10
3
relative MSE: AI vs AD
0 10 20 30 40
T
10
26
10
21
10
16
10
11
10
6
10
1
0 10 20 30 40
T
10
0
2 × 10
1
3 × 10
1
4 × 10
1
6 × 10
1
0 10 20 30 40
T
10
2
10
1
10
0
0 10 20 30 40
T
10
27
10
21
10
15
10
9
10
3
Varying n Varying T
GMM#1 median GMM#1 quantiles GMM#2 median GMM#2 quantiles GMM#3 median GMM#3 quantiles
Figure 5: Varying the number of samples n and the number of iterations T, we study the convergence
of EM, the local contractivity of F, and the MSEs of the OS and AI gradients against the AD gradient.
EM-MW2
2
-regularised generative model.
5.1 Barycentre Flow in 2D
Wasserstein barycentres [AC11] and their notoriously challenging computation [CD14; Álv+16;
AB22; TDG24] are active fields of research. In this section, we illustrate the use of differentiable EM
to flow a point cloud towards a barycentre of GMMs. Given M point clouds Yi ∈ R
n×2
, our goal is
to optimise a point cloud X ∈ R
n×2
, initialised as random normal noise, towards a barycentre (with
uniform weights) of GMMs (νi) fitted from (Yi). Specifically, we solve
min
X∈Rn×2
X
M
i=1
MW2
2

µ(F
T
X (θ0)), νi

with respect to X. The GMMs νi are fitted beforehand, and µ(F
T
X (θ0)) is the current EM estimate
of the optimised cloud X. We illustrate the results in Fig. 6, where M := 3, K := 2 and n := 500.
This method can be adapted to compute more general barycentres, as presented in Appendix A.7.2.
Figure 6: EM−MW2
2 flow displacing particles in order to make their EM output approach a barycentre
of three target GMMs.
5.2 Colour Transfer
Colour transfer is a well-known imaging task where OT techniques have been used extensively
[Rei+02; Del04; PK07; PKD07; PPC10; Rab+12; RFP14], it consists in transforming the RGB
15
Differentiable EM and OT S. Boïté, E. Tanguy, J. Delon, A. Desolneux and R. Flamary
colour distribution of a source image to match that of a target image. We propose the following
approach: we initialise an image X as the source image, and optimise it to minimise the MW2
2
cost
between a GMM fitted on X (seen as an RGB point cloud), and a target GMM ν ∈ GMM3(K) fitted
on the colour distribution of the target image. Specifically, we minimise X 7−→ MW2
2
(µ(F
T
X (θ0)), ν)
for some initialisation θ0. Warm-Start EM method from Algorithm 3, choose K := 10 components
and use fixed uniform GMM weights (Algorithm 2 for EM) to avoid being trapped in a local minimum, as seen in Section 3.3. We present some results in Fig. 7, and provide additional discussions
about the optimisation choices in Appendix A.7.3. Even with only K = 10 components, the colour
transfer preserves both details and global consistency, and in the resulting colour scheme, the source
image appears to exhibit stronger contrast in this example.
Balanced OT methods may be sensitive to outliers in the distributions, leading to artifacts in the
colour transfer results if colour aberrations are present in the target. Unbalanced OT methods
[LMS18] can mitigate this issue [Chi+18a; Bon+24]. We now consider the unbalanced variant of the
Mixture Loss defined in Section 3.4: by relaxing the constraints on marginals, it can ignore outliers
in the input distributions. Specifically, in Fig. 7, we consider an illustration with a corrupted target
image where a patch of red has been added, and observe that the unbalanced approach is more robust
to this aberration, showing no propagation of the red artifact.
Source Result Target
Balanced result Unbalanced result Corrupted target
Figure 7: Colour transfer experiments. Top row: EM − MW2
2
from source to target. Bottom row:
unbalanced colour transfer with regularisations λ1 = 10 (source) and λ2 = 0.1 (target).
5.3 Neural Style Transfer
We apply our distance within Gatys et al.’s neural style transfer framework [GEB16]. The goal is
to generate an image that combines the content of one image X0 with the artistic style of another
image Y . We rely on a pre-trained VGG-19 network [SZ15] (see Fig. 8a) to encode our image and
extract relevant features from three specific layers ℓ. Given an image X in R
3×H×W , we denote
these features VGG1···ℓ(X). Starting with the content image as X0, we optimise X such that its
features progressively match those of the style image Y . The target mixtures µ
style
ℓ
are fitted at the
beginning of the procedure on style features VGG1···ℓ(Y ). Notably, the target style is encoded as a
low-dimensional GMM, and the reference image is not needed during training, unlike in [GEB16].
Our objective is a weighted sum of Mixture Losses between the optimised features VGG1···ℓ(X) and
16
Differentiable EM and OT S. Boïté, E. Tanguy, J. Delon, A. Desolneux and R. Flamary
the target µ
style
ℓ
, for each layer ℓ in {1, 2, 3}: we solve
min
X∈R3×H×W
X
3
l=1
λℓ MW2
2

F(θinit, VGG1···ℓ(X)), µ
style
ℓ

. (23)
The weights λℓ
follow Gatys et al.’s scheme ([GEB15]) of 1/d2
ℓ where dℓ
is the dimension of features in
layer ℓ. We fit K = 3 Gaussian components at each layer. The chosen procedure for fitting Gaussian
mixtures is still Warm-Start EM. We optimise using 100 iterations of Adam with learning rate 0.01,
which takes approximately 20 seconds using CUDA on an RTX 4000. The example results shown in
Fig. 8 illustrate the ability of GMM to encode (and store) an image style which, to the best of our
knowledge, has never been shown. The experimental setup is further detailed in Appendix A.7.4,
along with additional examples.
(a) Model architecture (b) Source im. (c) Generated im. (d) Style im.
Figure 8: Style transfer method inspired by [GEB15]: setup and example result.
5.4 Texture Synthesis
We perform texture synthesis using a novel method inspired by [GLR18; LDD23; Hou+23]. We
initialise the synthesised texture using a stationary Gaussian field of the same mean and covariance
as the target texture. We then optimise a weighted sum of Mixture Losses over different scales in the
patch space (we refer the reader to Appendix A.7.5 for a full explanation). When doing multi-scale
synthesis, we simply choose to downscale the images by a factor 2
s
for s between 0 and S, so that
the image downscaled by a factor 2
S has size at least 16 × 16. In our experiments, we choose to fit
K = 4 Gaussian components in our mixtures. As illustrated in Fig. 9, this corresponds (roughly) to
the elbow of the model’s log-likelihood and gives more convincing results. As for patch sizes, notice
that choosing a patch size of 1 amounts to optimising the colour intensities directly, i.e. performing
standard colour transfer. In Fig. 10, we compare the results of taking 4 × 4 and 8 × 8 patches. The
mono-scale variant is functional for simple textures, as presented in Appendix A.7.5. A strength of
this approach is that it deals with multiple scales at once, whereas [LDD23; Hou+23] go through each
scale successively. This provides a simpler approach that is less sensitive to the transition method
between scales.
(a) K = 1 components (b) K = 4 components
1 2 3 4 5 6 7 8 9 10
K
0.95
1.00
1.05
log-likelihood
×106
(c) Log-likelihood of fit
Figure 9: Choosing the number of components K for texture synthesis.
17
Differentiable EM and OT S. Boïté, E. Tanguy, J. Delon, A. Desolneux and R. Flamary Reference Generated
Figure 10: Multi-scale texture synthesis with K = 4 components for 8 × 8 patches
Conclusion and Outlook
In this work, we have provided multiple strategies to differentiate the Expectation-Maximisation
algorithm, and illustrated their use in several applications. The overall practical message is that
when possible, the Warm-Start method is to be preferred, and when not applicable, the Automatic
Differentiation method is a good off-the-shelf solution. A core limitation of our methods (and EM to
a lesser extent) is the computational cost when the dimension is large: our applications were limited
to d ≈ 1000. For very high dimensional cases, one could consider sparse covariance representations
such as [BC25; Szw+25]. While we recommend using fixed uniform GMM weights, learnable weights
remain an option. Further numerical and algorithmic tweaks could be considered to avoid issues with
local optima. Finally, we focused on Gaussian Mixtures, but any mixture with differentiable densities
could be considered. Note that for a seamless extension, one still requires closed-form expressions
for the E and M steps.
Acknowledgements
This research was funded in part by the Agence nationale de la recherche (ANR), Grant ANR-23-
CE40-0017 and by the France 2030 program, with the reference ANR-23-PEIA-0004.
