Self-Attentive Spatial Adaptive Normalization for
Cross-Modality Domain Adaptation
Devavrat Tomar, Manana Lortkipanidze, Guillaume Vray
Behzad Bozorgtabar, and Jean-Philippe Thiran, Senior Member, IEEE
Abstract— Despite the successes of deep neural networks on many challenging vision tasks, they often fail
to generalize to new test domains that are not distributed
identically to the training data. The domain adaptation becomes more challenging for cross-modality medical data
with a notable domain shift. Given that specific annotated
imaging modalities may not be accessible nor complete.
Our proposed solution is based on the cross-modality
synthesis of medical images to reduce the costly annotation burden by radiologists and bridge the domain gap
in radiological images. We present a novel approach for
image-to-image translation in medical images, capable of
supervised or unsupervised (unpaired image data) setups.
Built upon adversarial training, we propose a learnable
self-attentive spatial normalization of the deep convolutional generator network’s intermediate activations. Unlike previous attention-based image-to-image translation
approaches, which are either domain-specific or require
distortion of the source domain’s structures, we unearth
the importance of the auxiliary semantic information to
handle the geometric changes and preserve anatomical
structures during image translation. We achieve superior
results for cross-modality segmentation between unpaired
MRI and CT data for multi-modality whole heart and multimodal brain tumor MRI (T1/T2) datasets compared to the
state-of-the-art methods. We also observe encouraging results in cross-modality conversion for paired MRI and CT
images on a brain dataset. Furthermore, a detailed analysis
of the cross-modality image translation, thorough ablation
studies confirm our proposed method’s efficacy.
Index Terms— Domain adaptation, image synthesis,
generative adversarial networks, unpaired domains, selfattention.
I. INTRODUCTION
D
IAGNOSTIC radiology encompasses diverse imaging
modalities for disease diagnosis and treatment. The most
popular medical imaging modalities used for radiotherapy
treatment planning and segmentation of tumor volumes are
magnetic resonance imaging (MRI) and computed tomography
(CT). MRI has been frequently used by radiologists to aid
D. Tomar and M. Lortkipanidze and G. Vray are with the Signal
Processing Laboratory (LT55), Ecole Polytechnique F ´ ed´ erale de Lau- ´
sanne (EPFL), Lausanne, Switzerland (e-mail: <devavrat.tomar@epfl.ch>;
<manana.lortkipanidze@epfl.ch>; <guillaume.vray@epfl.ch>). D. Tomar and
M. Lortkipanidze contributed equally to this work.
B. Bozorgtabar and J. Thiran are with the Signal Processing Laboratory (LT55), Ecole Polytechnique F ´ ed´ erale de Lausanne (EPFL), ´
Lausanne, Switzerland, and with the Radiology Department, Centre Hospitalier Universitaire Vaudois, Lausanne, Switzerland, and
also with the Centre d´Imagerie Biomedicale (CIBM) (e-mail: be- ´
<hzad.bozorgtabar@epfl.ch>; <jean-philippe.thiran@epfl.ch>).
(a) (b) (c) (d) (e)
Fig. 1: Examples of unsupervised cross-domain adaptation
between MRI/CT for cardiac substructure segmentation: (a)
original CT, (b) synthesized MRI, (c) segmentation result on
the CT image using a detached U-Net model trained on MRI
images, (d) segmentation result on the fake MRI (synthesized
from CT) using our proposed SASAN, and (e) ground-truth
segmentation.
organ-at-risk delineation due to its superior soft-tissue contrast [1]; however, compared with CT, it is relatively timeconsuming, expensive, and not readily accessible. On the other
hand, a CT scan uses X-ray beams to create fine slices [2],
which is a useful means of obtaining various levels of detail in
dense areas of the body. Nevertheless, CT scans cannot easily
distinguish soft tissue with poor contrast. As such, radiologists
may opt for an MRI. In practice, due to some limitations
such as inadequate scanning time, resources, and cost, certain
imaging modalities may be accessible nor complete.
These limitations could be overcome by Computer-Aided
Diagnosis (CAD) systems that can be deployed to impute the
missing image modalities by using other accessible modalities.
However, it has been pointed out that established machine
learning models would under-perform when tested on data
from different modalities due to the domain shift. This problem
is more pronounced as the domain shift between MRI/CT
data is remarkable such as the difference in the appearance of
tissues and anatomical structures. Therefore, there is a strong
clinical need to develop cross-modality image translation
systems to generalize the learned models from one modality
into another. These methods should ideally transfer knowledge
across image domains (modalities) without using additional
annotations from the target domain. Cross-modality image
translation is defined as learning a function f : X → Y that
maps images from source domain X to target domain Y by
using supervision (e.g. paired data) [3], [4] or by using unpaired data in an unsupervised setup [5], [6]. Image translation
Authorized licensed use limited to: Cornell University Library. Downloaded on May 24,2021 at 00:44:52 UTC from IEEE Xplore. Restrictions apply.
0278-0062 (c) 2021 IEEE. Personal use is permitted, but republication/redistribution requires IEEE permission. See <http://www.ieee.org/publications_standards/publications/rights/index.html> for more information.
This article has been accepted for publication in a future issue of this journal, but has not been fully edited. Content may change prior to final publication. Citation information: DOI 10.1109/TMI.2021.3059265, IEEE
Transactions on Medical Imaging
2 IEEE TRANSACTIONS ON MEDICAL IMAGING, VOL. XX, NO. XX, XXXX 2020
methods have gained attention due to the impressive success
of conditional generative models, e.g. conditional generative
adversarial networks (cGANs) [3], [4]. Currently, a large
portion of medical image synthesis approaches [7], [8] requires
a set of paired samples to build cross-modality reconstruction.
However, collecting a large number of aligned image pairs is
typically difficult, and sometimes even impractical to obtain.
Additionally, if the image pair examples from the two domains
are not available, finding such a mapping function is illposed as it involves estimating the joint distribution of the
two domains from the marginals. Nonetheless, if we assume
some prior over the joint-distribution and constraint the family
of mapping functions (e.g., similar colors, shapes, cyclic
consistency, consistent semantic layouts), we can estimate
the mapping function that best matches the given priors and
constraints. For example, the idea of the cyclic consistent generative adversarial network (CycleGAN) [5] performs image
translation using only the marginal distributions. However, for
a superior quality medical image modality translation, e.g., CTbased MR image construction, we require the organs’ anatomical structures to be preserved for detecting unhealthy tissues
and other anomalies in the synthesized translated image.
In this paper, we propose SASAN, short for the SelfAttentive Spatial Adaptive Normalization method, a new medical image modality conversion. We introduce a self-attention
module that attends to different anatomical structures of the
organ in the image to improve the image translation task. Our
method can leverage the auxiliary semantic segmentation information to guide the attention network and be used for paired
and unpaired datasets. Our cross-modality domain adaptation
has been further validated for medical image segmentation in
the absence of target domain labels. Examples of segmentation
results for cardiac substructures are shown in Fig. 1.
A. Contributions
Our contributions are as follows:
• We propose a new plug-and-play framework for unsupervised cross-modality adaptation of the medical image
segmentation. Our approach detaches the segmentation
network from the cross-modality adaptation process and
eliminates the target domain’s labeling cost. This is more
practical than the earlier approaches, which assume there
is access to a few amounts of annotated data on the target
domain or train the joint segmentation-domain adaptation
models,
• We propose a learnable self-attentive spatial normalization of the deep convolutional generator network’s
intermediate activations to preserve anatomical structures
during image translation,
• Our proposed attention regularization loss ensures learning orthogonal attention maps, each of them focusing
on specific anatomical regions and avoiding redundancy,
facilitating the translation process for different anatomical
structures. Besides, the proposed auxiliary semantic loss
term empowers our method to generate a missing modality with anatomical details, further guaranteeing the effectiveness of our approach towards clinical applications
and its generalizability,
• We demonstrate the effectiveness of our proposed
SASAN in tasks, including unsupervised image translation between multi-modal brain tumor MRI (T1/T2)
and cardiac substructure segmentation. The proposed
method achieves state-of-the-art performance on the
multi-modality whole heart segmentation task in qualitative and quantitative evaluations, making it a potential
solution to make accurate clinical decisions.
II. RELATED WORK
Traditional medical image synthesis methods are often
divided into three categories: learning-based methods [9]–
[11], tissue-segmentation-based methods [12], [13], and atlasbased methods [14]–[16]. However, there are certain pros
and cons of using these approaches. The learning-based approaches directly build a relationship between image domains
by mapping between feature representation of two different
domains. Therefore, the quality of mapping is greatly affected
by the feature selection process, which is not desirable. Tissuesegmentation based approaches, for instance, require manual
refinement of segmentation classes after the segmentation of
image voxels into a small number of tissue types. Furthermore,
atlas-based approaches use atlas images to align the image and
estimate matching pairs in another domain. However, finding
an atlas image is not straightforward and poses difficulties in
the workflow. Besides, it is hard to cover anatomical geometry
variations using atlas data.
The success of Convolutional Neural Networks (CNNs) for
the computer vision domain soon got adopted by researchers
for biomedical imaging tasks. It alleviates the need for
manual representation learning, as in learning-based classical
approaches, and offers automation of embeddings. However,
training CNNs requires a large amount of annotated data,
representing a considerable challenge in the biomedical domain. Besides, biomedical images are scarce due to patients’
privacy and require expert knowledge for proper annotation or
alignment. Recently, CNNs have been exploited for medical
image synthesis [17], achieving superior performance to the
classical machine-learning methods. For instance, Han et al.
[18] utilized CNN for image-modality conversion to synthesize
CT images from MR images using paired data. Similarly, Zhao
et al. [19] proposed a modified U-Net to synthesize an MR
image from a CT counterpart by minimizing voxel-wise differences between images. However, these approaches suffer from
the problem of blurred outputs caused by the optimization of
voxel-wise loss. Therefore, GANs have emerged as state-ofthe-art approaches for the domain for their realistic outputs,
including the recent variations conditional GAN, e.g., Pix2Pix
[3] and CycleGAN [5]. For example, Nie et al. [7] proposed a
context-aware GAN to synthesize CT images from input MR
images. Bi et al. [20] proposed a GAN-based method built
upon the pix2pix framework to synthesize positron emission
tomography (PET) images from CT images. There are other
concurrent GAN-based methods [21]–[24], which have been
proposed to learn a transformation in the pixel space for
medical synthesis. However, most of these methods use the
rigidly aligned cross-modality data and collecting the aligned
Authorized licensed use limited to: Cornell University Library. Downloaded on May 24,2021 at 00:44:52 UTC from IEEE Xplore. Restrictions apply.
0278-0062 (c) 2021 IEEE. Personal use is permitted, but republication/redistribution requires IEEE permission. See <http://www.ieee.org/publications_standards/publications/rights/index.html> for more information.
This article has been accepted for publication in a future issue of this journal, but has not been fully edited. Content may change prior to final publication. Citation information: DOI 10.1109/TMI.2021.3059265, IEEE
Transactions on Medical Imaging
TOMAR et al.: PREPARATION OF PAPERS FOR IEEE TRANSACTIONS ON MEDICAL IMAGING 3
paired training data is usually expensive. Especially the crossmodality data in medical institutions are often unpaired due
to restricted medical conditions. Some recent methods [6],
[25] alleviate this constraint and impose a cycle consistency
condition to use the unpaired data. For instance, Wolterink
et al. [25] applied a CycleGAN when synthesizing an MR
image using a CT image. However, weak-supervision without
additional constraints could lead to higher distortion and
inaccurate estimation of details, e.g., soft tissues in output
images. For example, the translated images could be heavily
skewed towards source image modalities in the CycleGAN.
Some other approaches [26], [27] use perceptual loss based
on similarity measure in feature space extracted from pretrained classification networks. Although these methods show
promising results and could generate high-quality synthetic
images, they do not incorporate semantic information, limiting
their capability to estimate the contextual information during
the cross-modality conversion.
All aforementioned GAN-based image translation methods
focus more on the overall appearance of the translated image
and have difficulty in generating fine structures. Recently,
some attention based approaches focus on translating foreground only [28] or distort the spatial context of the structures
[29]. For example, Kim et al. [29] proposed a method built
upon an attention mechanism’s idea to focus on the most
discriminative image regions that distinguish between source
and target domains during image translation. However, their
learned attention maps can deform the original structures
present in the domain source. This is usually beneficial for
specific datasets that require distortion of the source domain’s
structures for pleasing results. Nonetheless, for medical crossmodality translation, preserving the anatomical structures is
very crucial.
Some recent methods [30]–[32] have been proposed to
extend image synthesis to a general-purpose unsupervised
domain adaptation (UDA) framework. These methods aim to
bridge the gap between two domains at different representation
levels, such as feature [31], [33], [34] or pixel levels [35] or
both image and feature levels [32], [36] to validate further the
quality of the synthesis results for downstream tasks such as
segmentation. Qi et al. [32] proposed a plug-and-play domain
adaptation method that simultaneously matches feature and
output distribution across domains. However, the alignment
of feature-level marginal distributions does not explicitly enforce semantic coherence between modalities. Instead, we
address this issue via learning orthogonal attention maps,
each focusing on specific anatomical regions, facilitating the
translation process across image modalities. Hoffman et al.
[36] proposed CyCADA, an adversarial training approach that
benefits from both image-level and output-level adaptation
for the task of image segmentation. Cai et al. [37] utilized
a cycle consistent generative adversarial network for crossmodal organ translation and segmentation. Zhang et al. [38]
proposed a collaborative UDA method focusing on hard-totransfer samples to tackle label noise.
Another important track of domain adaptation for image
segmentation is based on self-training schema. Zou et al.
[39] proposed a UDA framework based on an iterative selftraining procedure for semantic segmentation. They also introduced a class-balanced self-training mechanism to address
the class imbalance issue for transfer learning. Xia et al.
[40] proposed an uncertainty-aware multi-view co-training
(UMCT) approach to utilize unlabeled data aimed at domain
adaptation and semi-supervised learning simultaneously for
3D volumetric image data.
III. METHODS
Upon the motivation mentioned in Section II for the
attention-based methods, this paper proposes an unsupervised
cross-modality domain adaptation for medical images to focus on preserving the geometrical relationship between the
anatomical structures during image translation by utilizing the
semantic information of image modalities.
For an unsupervised cross-modality adaptation, translating
different semantic regions between the two corresponding domains is often very important, especially for those biomedical
datasets. Different tissue regions should be translated differently from one domain to another for modality conversion between MRI and CT. To achieve this, we propose self-attentive
semantically learnable spatial feature normalization of the
activations of the generator model, as shown in Fig. 2. This
normalization technique is similar to [41]; however, instead of
using the segmentation maps for conditional image generation,
we use the proposed learned attention maps to normalize the
generator’s activations. Our generator can be divided into two
modules: synthesis module S and a mapping network A called
attention module. A learns the spatial semantic information as
attention maps, which are then fed to the decoder layers of S
to re-normalize their output activations based on the attended
information. We also utilize two variants of PatchGAN [5]
based discriminators; one focuses on the image translation
quality while the other focuses on the predicted segmentation
quality for the two domains. Using a separate discriminator
for predicting segmentation labels is motivated by the fact that
anatomical structures’ spatial relationships do not change for
the two domains’ images under image translation.
If the segmentation labels are provided e.g., for cardiac
structures segmentations [42], we use the segmentation labels to guide the attention mechanism for image-to-image
translation based on unpaired data and demonstrate how extra
information available can be utilized during training. For the
domain adaptation task, we omit the ground truth segmentation
labels of one domain (CT) during training and evaluate the
performance of a U-Net based MRI segmentation model on
the synthesized MRI. For the supervised image translation
task, e.g., on brain dataset [43], we also include few pairwise
pixel losses, as discussed in Section III-B, in addition to cycle
consistency assumption. As mentioned in [44], including cycle
consistency for paired cross-domain data helps to preserve the
structures while translating from one domain to another.
A. Self-Attentive Generator Architecture
As mentioned previously, our generator is composed of two
fully convolutional neural networks - synthesis module (S)
and attention module (A). A helps in transmitting the semantic
Authorized licensed use limited to: Cornell University Library. Downloaded on May 24,2021 at 00:44:52 UTC from IEEE Xplore. Restrictions apply.
0278-0062 (c) 2021 IEEE. Personal use is permitted, but republication/redistribution requires IEEE permission. See <http://www.ieee.org/publications_standards/publications/rights/index.html> for more information.
This article has been accepted for publication in a future issue of this journal, but has not been fully edited. Content may change prior to final publication. Citation information: DOI 10.1109/TMI.2021.3059265, IEEE
Transactions on Medical Imaging
4 IEEE TRANSACTIONS ON MEDICAL IMAGING, VOL. XX, NO. XX, XXXX 2020 ConvBlock 5×5 Stride 1, nf 32 ConvBlock 5×5, Stride 2, nf 64 ConvBlock 5×5, Stride 2, nf 128 ConvBlock 5×5, Stride 1, nf 128 ConvBlock 5×5, Stride 1, nf 128
Generator Encoder Generator Decoder SPADE ResnetBlock, nf 128 Upsample SPADE ResnetBlock, nf 64 Upsample SPADE ResnetBlock, nf 32
Conv 3×3,
Stride 1, nf 1
SPADE
ResnetBlock,
nf 32
Synthesis Module: S
Attention
Maps
Attention Module: A
SPADE
ResnetBlock
SPADE
LeakyRelu
Conv 3×3, Stride 1
SPADE
LeakyRelu
Conv 3×3, Stride 1
Attention
Maps
Input
Features
Output
Features
Conv 3×3
Relu
Instance
Norm
Conv 3×3 Conv 3×3
SPADE
Attention
Maps
Input
Features
Output
Features
(a) (b)
Fig. 2: Illustration of the proposed pipeline. (a) Generator architecture with a self-attention module, and (b) the SPADE
ResnetBlock architecture with the proposed self-attentive adaptive normalization. Encoder
Attention
Module B
Attention
Maps
Image B Fake A SPADE Decoder
Generator A
Encoder
Attention
Module A
SPADE
Decoder
Generator B
Image B Recon.
Train to Match
Train to Match
Discriminator
A
Predicted Seg B Predicted Seg Fake A
Encoder
Attention
Module A
Attention
Maps
Image A Fake B SPADE Decoder
Generator B
Encoder
Attention
Module B
SPADE
Decoder
Generator A
Image B Recon.
Train to Match
Train to Match
Discriminator
B
Predicted Seg A Predicted Seg Fake B
Train to Match
Ground Truth Seg A
Discriminator
Seg
Attention
Maps
Attention
Maps
Fig. 3: Overview of training. Domain adaptation between MRI (A) and CT (B). Attention Module extracts orthogonal
attention features that are fed to the decoder of the generator (SPADE Decoder) for image translation that works on three
different resolutions of the attention maps (64 × 64, 128 × 128, and 256 × 256). Since only the ground truth of domain A is
available, we train Attention Module B using an additional adversarial loss on the segmentation labels.
layout information to the intermediate layers of S, thus making
the translation task easier to learn. The architecture of S is
similar to the Pix2Pix generator architecture [5], except that
the Encoder is composed of Convolutional Blocks while the
Decoder contains SPADEResblock module proposed in [41]
that operates on different resolutions of attention maps derived
from the Attention Module (see Fig. 3).

1) Attention Module: The architecture of the attention module, A, is based on a lighter version of U-Net. We introduce
auxiliary losses and a regularization loss term as described in
Section III-B in the training objective to make the attention
maps semantically informative and orthogonal to each other.
2) Self-Attentive Spatial Adaptive Normalization: For normalizing the features of the decoder of the synthesis module,
the attention maps are first projected onto a smaller embedding
space and then convolved to produce affine parameters γ
and β, as shown in Fig. 2 (b). Unlike other conditional
normalization methods e.g., AdaIN [45] and AdaLIN [29], γ
and β are not vectors, but tensors with spatial dimensions. The
generated γ, and β are multiplied and added to the instance
normalized [46] input features give the final output features:
xˆtcij (m) = γtcij (m) ∗
xtcij − µtc
σtc

+ βtcij (m)
µtc =
1
HW
X
i,j
xtcij , σ2
tc =
1
HW
X
i,j
(xtcij − µtc)
2
where x represents the input features and xˆ(m) is the
normalized feature based on the attention maps m. µtc and
σ
2
tc are instance normalized mean and variance of the channel
c and batch t over the spatial location (i, j).
B. Loss Functions
Our model’s full objective comprises several loss functions,
some of which are included in the supervised setting when
ground-truth is available. We build our model based on the
GAN with a min-max objective. Given the images from
source domain X , we train a generator GY to transform the
images from source domain X to target-like images whose
visual appearances are similar to the real images from target
domain Y. At the same time, the attention module of GY
helps in preserving the original structural information. For
the unsupervised setting, the discriminators of the domain
Y, including the image discriminator DY and segmentation
Authorized licensed use limited to: Cornell University Library. Downloaded on May 24,2021 at 00:44:52 UTC from IEEE Xplore. Restrictions apply.
0278-0062 (c) 2021 IEEE. Personal use is permitted, but republication/redistribution requires IEEE permission. See <http://www.ieee.org/publications_standards/publications/rights/index.html> for more information.
This article has been accepted for publication in a future issue of this journal, but has not been fully edited. Content may change prior to final publication. Citation information: DOI 10.1109/TMI.2021.3059265, IEEE
Transactions on Medical Imaging
TOMAR et al.: PREPARATION OF PAPERS FOR IEEE TRANSACTIONS ON MEDICAL IMAGING 5
discriminator Dseg compete with the generator to correctly
differentiate the fake outputs from real ones. The attention
module for domain X is denoted as AX , which acts on the
images from domain X . Thus, the output of generator GY (x)
can be written as SY (x, AX (x)), where SY is the synthesis
module. We also include a single convolution layer AX on top
of the attention module AX to get the segmentation predictions
on the real and fake images of the domain X . Similarly, we
denote the discriminators of domain X as DX for images, the
generator from domain Y to X as GX and attention module
for domain Y as AY . In the supervised settings, we omit the
discriminator for segmentation if no segmentation labels are
available.

1) Adversarial Loss: We use an adversarial loss to match
the distribution of the translated images to the target image
distribution. We also employ adversarial loss for matching the
distribution of the predicted segmentation labels of translated
images and target images to the distribution of target segmentation labels. Using adversarial loss renders more realistic
outputs as the generator needs to trick discriminator and make
translated images look real. Therefore, the generator tries to
minimize the adversarial loss while the discriminator tries to
maximize it (detect translated images vs. real images) during
training. Instead of using vanilla GAN, we use Least Squares
GAN [47] objective for stable training:
L
X→Y
lsgan = L
Img
gan
X→Y

+ L
Seg
gan
X→Y (1)
where,
L
Img
gan
X→Y = Ey∼Y
DY (y)
2

+ Ex∼X
(1 − DY (GY (x))2

L
Seg
gan
X→Y = 2 ∗ Eyseg∼Yseg
Dseg(yseg)
2

+ Ey∼Y
(1 − Dseg(A
Y
(AY (y)))2

+ Ex∼X
(1 − Dseg(A
Y
(AY (GY (x)))))2

For domain adaptation, since only segmentation labels for
domain X are available, we replace yseg ∼ Yseg with xseg ∼
Xseg in the above equation as the distribution of the two
domains’ segmentation labels should be the same. The attention module AY also predicts the segmentation labels of the
domain Y, which are discriminated against the segmentation
labels of the domain X using the discriminator. This helps the
generator GX to generate fake images whose segmentation
labels are similar to real segmentation labels by enforcing
geometric relationships between the different structures.
2) Dual Cycle-Consistency Loss: Dual cycle-consistency
loss helps in avoiding model collapse during training. Given
an image x ∈ X , after the sequential translations of x from
domain X to Y and Y to X , the image must be reconstructed
back. The dual cycle-consistent loss and adversarial loss play
complementary roles. The former encourages a tight relationship between domains, while the latter helps generate realistic
images in an unsupervised training. To avoid over-fitting, we
furthermore use Structural Similarity Loss (SSIM) and L1 loss
for penalizing the cyclic reconstruction:
L
X→Y→X
cycle = Ex∼X
kGX (GY (x)) − xk1

+Ex∼X
1 − Lssim(GX (GY (x)), x)
 (2)
where,
Lssim(a, b) = X
i
(2µ
i
aµ
i
b + 1)(2σ
i
ab + 2)
(µi
a
2 + µ
i
b
2 + 1)(σ
i
a
2 + σ
i
b
2 + 2)
for some small 1 and 2. (µ
i
a
, µ
i
b
) and (σ
i
a
, σ
i
b
) represent
the means and standard deviations respectively of the i
th path
in images a and b.
3) Identity Loss: We also apply an identity consistency
constraint, which can regularize the generator to preserve the
colors and intensities during translation:
L
Y→Y
identity = Ey∼Y
kGY (y) − yk1

(3)
4) Attention Regularization Loss: We propose an attention
regularization loss term that encourages the attention maps to
be orthogonal with each other. This ensures that the attention
maps learn to attend to different regions in the image by
avoiding redundancy and increasing the translation power of
the model for different anatomical structures. The loss does
not require paired images and can be used in an unsupervised
manner omitting the need for extensive segmentation:
L
X
reg = Ex∼X
kAX (x)AX (x)
T − IkF

+Ey∼Y
kAX (GX (y))AX (GX (y))T − IkF

,
(4)
where I is the identity matrix and k.kF denotes the Frobenius norm.
5) Auxiliary Losses: We observed that training the attention
module A in a purely unsupervised manner is not trivial
and very time-consuming. Thus, to speed up the process
and improve the attention model’s power, we leverage all
information provided with the dataset. Therefore, we utilize
auxiliary losses that use the segmentation maps of the domain
(if available), making the multi-task model depending on
the data’s availability. Furthermore, we ensure that semantic
information present in the attention maps is consistent across
domains. To achieve this, we include one layer of convolution
over the attention maps to predict the segmentation labels and
compare them with the auxiliary ground-truth segmentation if
available using cross-entropy (CE) and Dice loss (DSC) on real
and fake images. For the case of domain adaptation, we replace
the ground-truth segmentation of the source domain (e.g., CT)
with pseudo labels derived from the attention module on the
fake image (e.g., fake MR):
L
X
aux = Ex∼X hX
c
CE(x
c
seg, AX
c
(AX (x)))
+
X
c
DSC(x
c
seg, AX
c
(AX (x)))i
+Ey∼Y hX
c
CE(y
c
seg, AX
c
(AX (GX (y))))
+
X
c
DSC(y
c
seg, AX
c
(AX (GX (y)))i
)
(5)
where,
Authorized licensed use limited to: Cornell University Library. Downloaded on May 24,2021 at 00:44:52 UTC from IEEE Xplore. Restrictions apply.
0278-0062 (c) 2021 IEEE. Personal use is permitted, but republication/redistribution requires IEEE permission. See <http://www.ieee.org/publications_standards/publications/rights/index.html> for more information.
This article has been accepted for publication in a future issue of this journal, but has not been fully edited. Content may change prior to final publication. Citation information: DOI 10.1109/TMI.2021.3059265, IEEE
Transactions on Medical Imaging
6 IEEE TRANSACTIONS ON MEDICAL IMAGING, VOL. XX, NO. XX, XXXX 2020
CE(a, b) = −
X
i,j
ai,j log(bi,j )
DSC(a, b) = 1 −
P
i,j 2ai,j bi,j
P
i,j ai,j + bi,j
(i, j) represent the pixel spatial location and AX
c
is one
convolutional layer classifier for segmentation class c.
6) Voxel-Wise Loss: If the paired dataset is available, we
derive l1 loss to impose a pixel-level penalty between the
translated image and the ground-truth. This loss is sensitive
to the alignment of images; thus, it can not be applied in an
unsupervised manner:
L
X→Y
voxel = Ex∼X ,y∼Y
kGY (x) − yk1

(6)
IV. EXPERIMENTS: UNSUPERVISED SETUP
The effectiveness of our proposed unsupervised crossdomain adaptation method is validated with applications on
MRI/CT images for the cardiac structure segmentation (see
Fig. 4) and MRI-T1/T2 for brain tumor segmentation. For
the latter task, even though we have access to the paired
images, we use them in an unsupervised manner. A U-Net
based segmentation model is detached from the learning of our
unsupervised domain adaptation to conduct a comprehensive
evaluation of different baselines. For each adaptation setting,
the ground-truths of target images were used for assessment
only, without being used during the training phase.
A. Datasets
For unpaired data, we evaluate our domain adaptation
method on MRI-CT whole heart segmentation dataset [42]
and MRI-T1/T2 brain tumor segmentation dataset [48] as
described below.

1) Whole-heart Segmentation Dataset [42]: This dataset
(MICCAI 2017 challenge) contains 20 MRI and 20 CT whole
cardiac images with accurate manual segmentation annotations
of ascending aorta (AA), left atrium blood cavity (LA-blood),
left ventricle blood cavity (LV-blood) and myocardium of the
left ventricle (LV-myo). We randomly split each modality of
the data into 80% training (16 subjects) and 20% testing (4
subjects) subsets for all experiments, so there is no subject
overlap ID among the subsets. For a fair comparison with SIFA
approach [30], we used the same preprocessing as the SIFA.
We first crop the central heart region for the pre-processing,
with four cardiac substructures selected for segmentation.
Then, for each 3D cropped image, we perform histogram
filtering between 2 and 98 percentile of pixel values followed
by normalization to zero-mean and unit standard deviation.
2) Multimodal Brain Tumor Segmentation [49]: The dataset
(BRATS, 2015) contains 65 multi-contrast MR scans from
glioma patients and consists of four different contrasts - T1,
T1c, T2, and FLAIR. Furthermore, expert annotations are
given for “edema,” “non-enhancing (solid) core,” “necrotic (or
fluid-filled) core,” and “enhancing core” segmentation classes.
For the pre-processing, the dataset was co-registered for each
subject’s image volumes to the T1c MRI, since it had the
highest spatial resolution. Additionally, we clipped 2 and 98
percentiles to remove outliers. We split the data randomly at
the subject-level into on-overlapping training (90%), and test
(10%) sets.
Data Augmentation. We use the same data augmentation
for all datasets. We perform transformation operations, such
as random rotation, random width, and height scaling [0.75,
1], flipping an image horizontally/vertically.
B. Experimental Setup
For unsupervised settings, we train the two generators
(GX , GY ) and discriminators (DX , DXseg
, DY , DYseg
) jointly
in a cyclic adversarial manner by optimizing the following
objective:
min
GY ,GX ,AY
c
,AX
c
max
DY ,DX
L
X→Y
un paired + L
Y→X
un paired (7)
where,
L
X→Y
un paired = L
X→Y
lsgan + λcL
X→Y→X
cycle + λidL
Y→Y
identity
+λregL
X
reg + λauxL
X
aux
We alternatively update the generators and discriminator in
one step using Adam optimizer [50] with parameters β1 = 0.5,
β2 = 0.999 and learning rate of 10−4
for the first 50 epochs
which is linearly decreased to 0 for the next 50 epochs.
Fig. 6 shows qualitative comparison of image translation of
our method with CycleGAN [5] and U-GAT-IT [29]. For
evaluating the methods quantitatively, we train a separate UNet based segmentation model to segment the four structures:
AA, LA-blood, LV-blood, and LV-myo, and compare the
segmentation accuracy on the translated images. Fig. 4 shows
the qualitative comparison of segmentation results between our
SASAN and other methods for CT to MRI image translation,
while the quantitative evaluations are shown in Table I and
Table II for both adaptation directions. Our code is available
on GitHub.1
C. Performance Metrics
We employ the widely used evaluation protocols, the average symmetric surface distance (ASSD) and the Dice similarity coefficient (Dice), to quantitatively measure the performance of domain adaptation models for the segmentation
task. In particular, we run a segmentation model on the
synthesized medical images to see how well the predicted
segmentation map matches ground-truth. The evaluation is
conducted based on the subject-level segmentation volume.
A higher Dice value and a lower ASSD value indicate more
substantial capabilities of adaptation models to preserve the
organs’ anatomical structures and generate realistic images. It
should also be noted that if one of the segmentation labels is
missing, the ASSD values become infinity. In this case, we
upper bound the ASSD score to 50.
1<https://github.com/devavratTomar/sasan>
Authorized licensed use limited to: Cornell University Library. Downloaded on May 24,2021 at 00:44:52 UTC from IEEE Xplore. Restrictions apply.
0278-0062 (c) 2021 IEEE. Personal use is permitted, but republication/redistribution requires IEEE permission. See <http://www.ieee.org/publications_standards/publications/rights/index.html> for more information.
This article has been accepted for publication in a future issue of this journal, but has not been fully edited. Content may change prior to final publication. Citation information: DOI 10.1109/TMI.2021.3059265, IEEE
Transactions on Medical Imaging
TOMAR et al.: PREPARATION OF PAPERS FOR IEEE TRANSACTIONS ON MEDICAL IMAGING 7
(a) (b) (c) (d) (e) (f)
Fig. 4: Visual comparison of segmentation results produced by different domain adaptation methods between MRI/CT data for
the cardiac substructure segmentation task: for each baseline, we evaluate the segmentation accuracy of generated MRI from
CT using a detached U-Net segmentation model that was trained on real MRI: (a) original CT, (b) results without domain
adaptation, (c) Cycle-GAN (d) U-GAT-IT, (e) SASAN (ours), and (f) ground-truth. The structures of AA, LA-blood, LV-blood
and LV-myo are highlighted by red, green, blue and yellow colors, respectively. Best viewed in color.
TABLE I: Performance comparison with different unsupervised cross-domain (MRI → CT) adaptation methods for heart
structures segmentation. MR-Seg and CT-Seg represent trained U-Net model for MRI and CT domains, which are fine-tuned,
and the segmentation results are reported volume-wise. (*implies as reported in the published paper.) We get an overall 2%
improvement in average Dice score compared to SIFA (p-value<0.3 for Welch’s t-test on slices.)
Method AA LA-blood LV-blood LV-myo Mean
Dice ASSD Dice ASSD Dice ASSD Dice ASSD Dice ASSD
MR-Seg 0.84±0.05 3.6±2.3 0.86±0.08 2.2±1.1 0.92±0.03 2.9±2.3 0.79±0.03 2.8±1.8 0.85±0.04 2.9±0.5
CT-Seg 0.86±0.17 5.5±4.0 0.91±0.02 5.2±0.9 0.92±0.02 3.6±2.3 0.86±0.03 5.8±3.7 0.89±0.03 5.0±0.8
W/o adaptation 0.23±0.15 41.1±20.5 0.13±0.13 30.9±18.6 0.00±0.00 N.A. 0.01±0.01 35.0±6.7 0.09±0.09 N.A.
CycleGAN 0.70±0.07 13.6±2.8 0.69±0.06 11.6±3.8 0.52±0.20 9.3±3.9 0.29±0.13 8.8±4.2 0.55±0.13 10.9±3.8
U-GAT-IT 0.68±0.08 12.0±3.4 0.66±0.08 13.7±4.1 0.55±0.16 8.9±3.5 0.39±0.12 8.9±3.3 0.57±0.11 10.9±3.5
Pnp-Ada-Net* 0.74±0.07 12.8±3.2 0.68±0.05 6.3±2.3 0.62±0.11 17.4±7.0 0.51±0.07 14.7±4.8 0.64±0.08 12.8±4.3
SIFA 0.84±0.05 7.01±2.70 0.84±0.04 3.8±1.1 0.72±0.14 4.82±2.33 0.62±0.16 4.73±1.61 0.76±0.08 5.1±1.4
SASAN (final) 0.82±0.02 4.14±1.6 0.76±0.25 8.3±2.2 0.82±0.03 3.5±1.1 0.72±0.08 3.3±0.9 0.78±0.10 4.9±1.5
TABLE II: Performance comparison with different unsupervised cross-domain (CT → MRI) adaptation methods for heart
structures segmentation. Our method beats the second best model SIFA on average Dice score by a margin on 5% (p-value <
0.2 for Welch’s t-test on slices.)
Method AA LA-blood LV-blood LV-myo Mean
Dice ASSD Dice ASSD Dice ASSD Dice ASSD Dice ASSD
W/o adaptation 0.01 ±0.01 45.0±28.1 0.04±0.07 40.2 ± 10.5 0.28 ±0.19 17.0 ± 7.3 0.01±0.01 27.9±6.8 0.09 ±0.11 32.5±10.9
CycleGAN 0.53 ±0.14 17.9±4.3 0.41±0.08 13.3±4.4 0.65±0.13 8.7±5.3 0.44±0.09 6.7±3.2 0.51 ±0.14 11.7±6.0
U-GAT-IT 0.55 ± 0.15 16.5±4.4 0.39±0.10 12.1±3.1 0.69±0.11 7.63±5.2 0.49 ± 0.08 7.0 ±3.3 0.53±0.13 11.0 ± 5.3
Pnp-Ada-Net* 0.44±0.11 11.4±3.2 0.47±0.07 14.5±4.1 0.78±0.10 4.5±1.4 0.49±0.03 5.3±1.8 0.54±0.08 8.9±2.6
SIFA 0.70±0.04 5.0±2.3 0.65±0.07 7.9±2.8 0.78±0.03 4.1±0.7 0.45±0.03 4.4±0.70 0.65±0.01 5.4±1.0
SASAN (low Res.) 0.34±0.09 23.3 ±6.4 0.58±0.09 12.9±2.6 0.76±0.07 6.8±2.1 0.46±0.09 9.5±1.6 0.53±0.06 13.1±1.7
SASAN (16 Att.) 0.41±0.10 23.4 ±5.0 0.64±0.10 11.1±2.7 0.84±0.06 5.9±3.4 0.64±0.04 3.7±1.6 0.63±0.05 11.1±5.7
SASAN (final) 0.54±0.09 18.8 ±5.0 0.73±0.06 9.4±3.0 0.86±0.09 6.1±4.3 0.68 ± 0.08 3.85±1.7 0.70±0.11 9.5±3.2
Authorized licensed use limited to: Cornell University Library. Downloaded on May 24,2021 at 00:44:52 UTC from IEEE Xplore. Restrictions apply.
0278-0062 (c) 2021 IEEE. Personal use is permitted, but republication/redistribution requires IEEE permission. See <http://www.ieee.org/publications_standards/publications/rights/index.html> for more information.
This article has been accepted for publication in a future issue of this journal, but has not been fully edited. Content may change prior to final publication. Citation information: DOI 10.1109/TMI.2021.3059265, IEEE
Transactions on Medical Imaging
8 IEEE TRANSACTIONS ON MEDICAL IMAGING, VOL. XX, NO. XX, XXXX 2020
(a) Attention maps on CT images with Lreg
(b) Attention maps on CT images without Lreg
Fig. 5: The visualizations of the attention maps for CT cardiac substructures for MRI → CT domain adaptation. Left to right:
CT image, corresponding 8 attention maps.
(a) (b) (c) (d)
Fig. 6: Qualitative results of different image translation methods from CT to MRI. From left to right: (a) input CT,
the translation results of (b) Cycle-GAN, (c) U-GAT-IT, (d)
SASAN.
D. Unsupervised Domain Adaptation
We evaluate the effectiveness of the proposed method for
the cross-domain image synthesis via measuring segmentation
accuracy on the fake MRI (synthesized from CT) using a
detached U-Net model trained on real MRI. We also observed an improvement in the segmentation accuracy when
the detached U-Net is trained with real MRI along with the
fake MRI images and the corresponding fake segmentation
labels generated by the Attention Module as shown in Table III. We first provide the performance upper bound of
supervised training and then obtain the performance lower
bound “W/o adaptation” by directly applying the segmentation
model learned in the source domain to test target images
without using any cross-domain adaptation method. For a
fair comparison, we utilize the same network architecture
and experimental setup for all baselines experiments. Table
I reports the segmentation results of unsupervised domain
adaptation (MRI → CT) methods for the cardiac dataset. As
shown in Table I, when directly applying the trained segmenter
on MRI images to test data (CT images), we obtain the Dice
score of 0.09±0.09, indicating that severe domain gap would
severely impede the generalization ability of deep models
Authorized licensed use limited to: Cornell University Library. Downloaded on May 24,2021 at 00:44:52 UTC from IEEE Xplore. Restrictions apply.
0278-0062 (c) 2021 IEEE. Personal use is permitted, but republication/redistribution requires IEEE permission. See <http://www.ieee.org/publications_standards/publications/rights/index.html> for more information.
This article has been accepted for publication in a future issue of this journal, but has not been fully edited. Content may change prior to final publication. Citation information: DOI 10.1109/TMI.2021.3059265, IEEE
Transactions on Medical Imaging
TOMAR et al.: PREPARATION OF PAPERS FOR IEEE TRANSACTIONS ON MEDICAL IMAGING 9
when compared to the performance of upper bound (the Dice
score of 0.85±0.04). Our proposed SASAN outperforms stateof-the-art domain adaptation methods, consistently providing
improvements across different modalities for the segmentation
performance. For the synthesized MRI from CT images, we
improved the Dice score to 0.78±0.07 over the four cardiac
segmentation classes, with the ASSD score being decreased
to 4.9±1.5. In addition, the domain adaptation results for the
reverting direction (CT → MRI) are shown in a Table II.
Overall, it is difficult to locate the cardiac substructures for
the reverting direction and in the synthesized CT from MRI
image because of its limited contrast with the surrounding
tissue. The qualitative segmentation results in Fig. 4 show
that it is challenging to delineate cardiac structure without
domain adaptation. Instead, our method enables a flexible
adaptation of the segmentation of four cardiac structures. Besides, anatomy-consistency is encouraged with our approach,
yielding satisfactory segmentation results of different organs
with varying shapes and sizes. Furthermore, We show the
visualization of attention maps for the cardiac substructures
in Fig. 5. As indicated by qualitative results, the proposed
Attention regularization loss term ensures learning orthogonal
attention maps, thus focusing on different anatomical regions
and facilitating the translation process across image modalities.
E. Comparison With State-of-The-Art Methods
We compare our method with three leading unsupervised
domain adaptation approaches for the cardiac segmentation:
the PnP-Ada-Net model [32], the CycleGAN model [5], and
the U-GAT-IT model [29], respectively. The U-GAT-IT [29] is
the current state-of-the-art attention-based method for image
synthesis. The four cardiac structures (categories), including
AA, LA-blood, LV-blood, and LVmyo, are used for segmentation performance evaluation. We summarize the mean±std
of ASSD and Dice metrics in Table I (MRI→CT) and Table
II (CT→MRI). Overall, our SASAN achieves higher Dice
score over the non-attention cross-domain adaptation baseline
(CycleGAN). Besides, our method surpasses the state-of-theart attention-based method (U-GAT-IT) by a large margin of
0.21 in Dice and 6.0 in ASSD, respectively. Moreover, as
shown in Fig. 4, translated results using SASAN are visually
superior to other methods, e.g., U-GAT-IT, while preserving
the source domain’s anatomical structures. This highlights
the importance of our proposed self-attentive spatial feature
normalization, facilitating translation of various tissue regions
differently from one domain to another without distortion of
the source domain’s structures. Finally, except for LA-blood,
SASAN performs favorably against one of the state-of-the-art
frameworks PnP-Ada-Net. Similar to our approach, PnP-AdaNet utilized domain adaptation based on output segmentation.
However, compared with PnP-Ada-Net, segmentation results
generally get improved with our self-attention module that
attends to different anatomical structures of the organ, enabling
tissue-specific translation.
We also compare the proposed SASAN with SIFA [30]
using Welch’s t-test for the null hypothesis: the average Dice
scores are the same for the two methods. As mentioned in
Table I, for MRI to CT domain adaptation, we get the p-value
< 0.3, implying we can reject the above hypothesis with a
confidence probability >0.7. Similarly, in table II, we get the
p-value<0.2 for CT to MRI domain adaptation. We achieve
marginal improvement compared to the SIFA method [30] in
terms of average Dice score.
F. Ablation Studies
Efficacy of Output Segmentation-Level Adaptation. As
mentioned previously, to further address the domain shift between image modalities, we build an additional convolutional
discriminator on top of the auxiliary prediction level obtained
by the attention module. We investigate the effectiveness of
this output-level adaptation using an ablation experiment for
the segmentation task. To do so, we turn-off the adversarial
loss term on segmentation by removing the second discriminator and then generating the baseline results for evaluation. We
compare this variant (SASAN w/o Dseg) to our final model,
which aligns the joint space of cardiac structures segmentation
outputs. As shown in Table III, output space adaptation on
cardiac segmentation prediction achieves performance gain
of about 0.08 in average Dice score and 3.4 average ASSD
score. This indicates that coupling the output-level adaptation
(segmentation labels) with the input image-level adaptation
improves the performance of segmentation.
Efficacy of Auxiliary Loss. We also assess the effects of
segmentation loss for the real source domain images (MRI)
and the synthetic target domain images (synthetic CT) generated during training. Even though we have segmentation
labels for MRI images only, we used the same segmentation
labels for the synthetic CT images as they were generated
from MRI and thus should have the same semantic layout. As
shown in Table III, using these pseudo segmentation labels
help the attention module of the CT domain to focus on
different cardiac structures of the real CT images.
Efficacy of Attention Maps Orthogonality. As evident in
Table III, we observe that removing redundancy of information
from the attention maps is helpful to get better domain
adaptation results. We conduct ablation experiments with and
without attention regularization loss (SASAN w/o Lreg) as
described in Section III-B and observe an overall improvement
of 9% in average Dice score. We also increased the number of
attention maps from 8 to 16 and reported the results for each
substructure in Table III. Increasing the number of attention
maps may add redundancy and decrease performance. We also
observe that sometimes without regularization on the attention
maps, the attention module tends to focus on one cardiac
substructure (LA-blood), leading to slight improvement. However, the overall segmentation accuracy without regularization
decreases.
Sensitivity Test for the Image Resolutions. We also repeated the experiment using different image scaling to evaluate
the performance’s influence, particularly with an input image
resolution of 128 × 128. Overall, as shown in Table III, the
results demonstrate, increasing image resolution yields better
image translation and higher adaptation performance based on
segmentation accuracy.
Authorized licensed use limited to: Cornell University Library. Downloaded on May 24,2021 at 00:44:52 UTC from IEEE Xplore. Restrictions apply.
0278-0062 (c) 2021 IEEE. Personal use is permitted, but republication/redistribution requires IEEE permission. See <http://www.ieee.org/publications_standards/publications/rights/index.html> for more information.
This article has been accepted for publication in a future issue of this journal, but has not been fully edited. Content may change prior to final publication. Citation information: DOI 10.1109/TMI.2021.3059265, IEEE
Transactions on Medical Imaging
10 IEEE TRANSACTIONS ON MEDICAL IMAGING, VOL. XX, NO. XX, XXXX 2020
TABLE III: Ablation studies for MRI to CT domain adaptation for cardiac substructures segmentation. Abbreviations - w/:
with, w/o: without, aug.: data augmentation using fake images and fake labels, Att.: Number of Attention maps, low res.:
image resolution of 128×128.
Method AA LA-blood LV-blood LV-myo Mean
Dice ASSD Dice ASSD Dice ASSD Dice ASSD Dice ASSD
SASAN w/ aug. 0.82±0.02 4.14±1.6 0.76±0.25 8.3±2.2 0.82±0.03 3.5±1.1 0.72±0.08 3.3±0.9 0.78±0.10 4.9±1.5
SASAN w/o aug. 0.70±0.09 20.3±2.4 0.72±0.11 14.6±5.3 0.74±0.11 7.5±2.4 0.68±0.11 7.9±3.9 0.71±0.07 12.6±2.0
SASAN w/o Dseg 0.63±0.11 22.3±2.7 0.57±0.16 27.3±14.9 0.71 ±0.04 8.2±2.9 0.61±0.10 6.5±2.0 0.63±0.05 16.0±8.9
SASAN w/o Laux 0.60±0.11 24.2±3.3 0.55±0.15 26.0±14.1 0.62±0.04 8.9±3.0 0.53±0.10 6.5±2.4 0.58±0.04 16.4±8.8
SASAN w/o Lreg 0.61±0.04 24.9±0.9 0.81±0.05 11.1±5.3 0.55±0.08 7.6±2.4 0.53±0.16 10.0±4.9 0.62±0.11 13.5±6.8
SASAN w/ 16 Att. 0.59±0.12 22.3±3.6 0.58±0.03 23.4±7.3 0.54±0.13 7.9±1.3 0.45±0.16 10.6±2.8 0.54±0.06 16.0±6.9
SASAN low res. 0.48±0.14 14.10±1.44 0.73±0.03 11.31±2.97 0.55±0.23 4.66±2.70 0.52±0.22 6.14±1.15 0.57±0.10 9.05±3.82
SASAN Lreg = 5 0.68±0.12 20.58±2.87 0.67±0.15 20.44±10.42 0.58±0.20 8.37±2.78 0.64±0.15 9.83±2.41 0.64±0.04 14.80±5.73
SASAN Laux = 6. 0.67±0.10 21.24±1.68 0.81±0.05 10.68±3.18 0.61±0.17 6.96±3.62 0.66±0.17 6.09±1.98 0.69±0.07 11.24±6.02
TABLE IV: Performance comparison for unsupervised crossdomain (MRI-T2 → MRI-T1) adaptation of brain segmentation. Abbreviations- Necrosis: Necrotic (fluid-filled) core, NE
Tumor: Non-enhancing (solid) core, E Tumor: enhancing core.
Method Necrosis Edema NE Tumor E Tumor Mean
CycleGAN 0.39±0.10 0.55±0.08 0.13±0.11 0.40±0.07 0.38±0.09
SASAN (Ours) 0.44±0.08 0.61±0.07 0.18±0.06 0.46±0.07 0.42±0.07
Upper-bound 0.50±0.07 0.65±0.06 0.35±0.05 0.51±0.06 0.50±0.06
G. Efficacy of Image Synthesis for Data Augmentation
We show that the use of our SASAN leads to realistic
synthetic images that contribute to improving the segmentation
performance if synthetic images are used alongside the real
images to fine tune the model. In particular, we observed an
improvement for the upper bound U-Net model’s accuracy
by performing data augmentation with synthetic MRI images
and their corresponding ground-truth segmentation labels from
the CT domain. Besides, the synthetically generated images
can alleviate the common problem of class imbalance despite
having a tiny number of real training samples, thus resulting
in a more robust system. Table V shows the performance
gain of SASAN in the mean Dice score to 0.87 using data
augmentation with fake MRI when compared to the average
Dice score of 0.85 without data augmentation.
Moreover, we also observe an improvement (as shown in
Table III) in the mean Dice and ASSD scores when the
detached U-Net is trained on the real MRI data with the
ground-truth segmentation labels and the synthetic MRI and
synthetic labels generated by our method using the generator
and attention modules.
H. Domain Adaptation for Brain Tumor Segmentation
Our unsupervised domain adaptation method can also faithfully translate brain tumor regions from MRI-T1 modality to
MRI-T2 modality, as depicted in Fig. 7. As the T2 domain
has a different contrast, directly feeding T1 brain images to
the T2 segmentation model gives poor results with an average
Dice score of just 0.02 for the four tumor classes. On the other
hand, our T2 to T1 domain adaptation method increases the
average Dice score to 0.42. Furthermore, we add a comparison
with an additional baseline method, CycleGAN, and report the
upper-bound for the brain segmentation classes in Table IV.
(a) (b) (c) (d) (e)
Fig. 7: Visual comparison of segmentation results produced
by domain adaptation from MRI-T2 to MRI-T1: (a) original
MRI-T1, (b) fake MRI-T2 (synthesized from MRI-T1) , (c)
segmentation results on MR-T1 image using a detached U-Net
model trained on real MRI-T2 images, (d) segmentation results
with domain adaptation, and (e) ground-truth. We obtain the
average Dice score of 0.42±0.07 with domain adaptation
compared to the Dice score of only 0.02±0.06 without domain
adaptation.
TABLE V: Improvement in MRI segmentation accuracy for a
model trained on available training MRI data and additional
synthetic CT data (CT to MRI translation) with ground-truth.
Cardiac
Substructure
Dice w/o
Data Augmentation
Dice with
Data Augmentation
AA 0.84±0.05 0.87±0.05
LA-Blood 0.86±0.08 0.86±0.07
LV-Blood 0.92±0.03 0.91±0.03
LV-myo 0.79±0.03 0.82±0.03
Mean 0.85±0.04 0.87±0.04
Authorized licensed use limited to: Cornell University Library. Downloaded on May 24,2021 at 00:44:52 UTC from IEEE Xplore. Restrictions apply.
0278-0062 (c) 2021 IEEE. Personal use is permitted, but republication/redistribution requires IEEE permission. See <http://www.ieee.org/publications_standards/publications/rights/index.html> for more information.
This article has been accepted for publication in a future issue of this journal, but has not been fully edited. Content may change prior to final publication. Citation information: DOI 10.1109/TMI.2021.3059265, IEEE
Transactions on Medical Imaging
TOMAR et al.: PREPARATION OF PAPERS FOR IEEE TRANSACTIONS ON MEDICAL IMAGING 11
V. EXPERIMENTS: SUPERVISED SETUP
A. Dataset
We assess the efficacy of our method for supervised modality conversion for the CT-MR brain dataset [43].
3) RIRE Dataset [43]: This dataset consists of paired but unregistered CT and MR images of 20 patients. We first register
the same patient’s CT and MRI volumes by maximizing their
joint histogram mutual information. Furthermore, we remove
the outlier noise by clipping the pixel values between 0.5 and
99.5 percentiles and also mask out the outer circle noise from
CT images. We split the data randomly at the patient-level into
on-overlapping training (18 patients), and test (2 patients) sets.
This section presents experimental results of supervised
training for image translation on MRI/CT brain images from
the RIRE dataset. Our proposed SASAN has been evaluated
alongside the baseline supervised methods Pix2Pix, MR-GAN,
and DC2Anet. Since the segmentation labels of the RIRE
dataset are not available at training, learned attention results
are even more critical since they allow us to spare segmentation annotations and capture different complex relationships
between tissues/parts of the brain. Fig. 9 shows sample four
attention maps produced by SASAN. It is noticeable that the
attention module manages to distinguish between different
regions such as inner region, outer frontal contour, outer
back contour, etc. This part can also be seen as automation
of methods similar to tissue-segmentation based classical
approaches. SASAN does not require heavy and expensive
annotations of segmentation. However, it manages to refine
different regions based on learned attention maps. Regarding
the relative performance, the attention module allowed us to
improve upon other prominent approaches, as shown in Table
VI.
For this, we conducted an evaluation study using the distortion metrics, e.g., the Structural Similarity Index (SSIM),
the Peak Signal to Noise Ratio (PSNR), Mean Absolute Error
(MAE), Root-Mean-Square Error (RMSE), Pearson Correlation Coefficient (PCC) as well as the learned perceptual image
patch similarity (LPIPS) [51] for the supervised setting. The
purpose of this experiment is to evaluate how synthesized MRI
images recover realistic textures faithful to bone/tissue regions.
As shown in Table VI, Our SASAN achieves competitive
performance with respect to supervised image translationbased methods. However, as shown in [52], [53], distortion
metrics such as the PSNR used as quantitative metrics do
not match up directly with perceptual quality. For example,
the synthesized MRI images using both our SASAN and
the DC2Anet are not ranked first in terms of the PSNR or
MAE metrics; however, they generate more visually appealing results in Fig. 8 than the other methods, and they can
preserve the underlying geometry and structure of anatomical
regions. This shows leveraging the geometric cues; our model
produces realistic textures and sharper edges for synthesized
MRI images, which are not easily visible on CT images. In
addition, we added an error map between the generated image
and the reference image for each baseline method
TABLE VI: Performance comparison with different supervised
image translation methods (CT → MRI) on the RIRE dataset.
LPIPS MAE PSNR SSIM RMSE PCC
Pix2Pix 0.06±0.01 0.02±0.01 22.42±0.97 0.87±0.01 0.08±0.01 0.96±0.01
MR-GAN 0.05±0.01 0.03±0.01 21.46±0.74 0.90±0.02 0.08±0.01 0.97±0.01
DC2Anet 0.05±0.01 0.04±0.01 20.99±0.98 0.90±0.01 0.09±0.01 0.97±0.01
SASAN (Ours) 0.05±0.01 0.04±0.01 21.37±1.29 0.90±0.02 0.09±0.01 0.97±0.01
VI. CONCLUSION AND FUTURE WORK
This paper proposed a generic domain adaptation framework, SASAN, to synthesize missing modality for biomedical
images and improve the generalization ability of CNNs across
different image modalities. The proposed approach is based
on a self-attention mechanism that attends various anatomical
structures of the organ in the image to improve cross-domain
image synthesis. Besides, our proposed auxiliary loss can be
used in a plug-and-play way to leverage semantic information
and guide the attention network, thus facilitating the transformation. Promising results in brain image synthesis and cardiac
segmentation show the effectiveness of our approach. Further
ablation studies have validated that SASAN improves existing
synthesis methods in quantitative and qualitative measures,
establishing new state-of-the-art.
The current limitation of unsupervised domain adaptation
methods is they need a fully labeled source domain, which
hinders their applicability in the practical clinical workflow.
A possible future extension of this work would be utilizing
self-supervised learning to leverage unlabeled data, yielding
more practical domain adaptation, where only a small fraction
of data in source domain is annotated.
