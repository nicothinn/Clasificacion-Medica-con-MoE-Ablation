AN IMAGE IS WORTH 16X16 WORDS:
TRANSFORMERS FOR IMAGE RECOGNITION AT SCALE
Alexey Dosovitskiy∗,†
, Lucas Beyer∗
, Alexander Kolesnikov∗
, Dirk Weissenborn∗
,
Xiaohua Zhai∗
, Thomas Unterthiner, Mostafa Dehghani, Matthias Minderer,
Georg Heigold, Sylvain Gelly, Jakob Uszkoreit, Neil Houlsby∗,†
∗
equal technical contribution, †
equal advising
Google Research, Brain Team
{adosovitskiy, neilhoulsby}@google.com
ABSTRACT
While the Transformer architecture has become the de-facto standard for natural
language processing tasks, its applications to computer vision remain limited. In
vision, attention is either applied in conjunction with convolutional networks, or
used to replace certain components of convolutional networks while keeping their
overall structure in place. We show that this reliance on CNNs is not necessary
and a pure transformer applied directly to sequences of image patches can perform
very well on image classification tasks. When pre-trained on large amounts of
data and transferred to multiple mid-sized or small image recognition benchmarks
(ImageNet, CIFAR-100, VTAB, etc.), Vision Transformer (ViT) attains excellent
results compared to state-of-the-art convolutional networks while requiring substantially fewer computational resources to train.1
1 INTRODUCTION
Self-attention-based architectures, in particular Transformers (Vaswani et al., 2017), have become
the model of choice in natural language processing (NLP). The dominant approach is to pre-train on
a large text corpus and then fine-tune on a smaller task-specific dataset (Devlin et al., 2019). Thanks
to Transformers’ computational efficiency and scalability, it has become possible to train models of
unprecedented size, with over 100B parameters (Brown et al., 2020; Lepikhin et al., 2020). With the
models and datasets growing, there is still no sign of saturating performance.
In computer vision, however, convolutional architectures remain dominant (LeCun et al., 1989;
Krizhevsky et al., 2012; He et al., 2016). Inspired by NLP successes, multiple works try combining
CNN-like architectures with self-attention (Wang et al., 2018; Carion et al., 2020), some replacing
the convolutions entirely (Ramachandran et al., 2019; Wang et al., 2020a). The latter models, while
theoretically efficient, have not yet been scaled effectively on modern hardware accelerators due to
the use of specialized attention patterns. Therefore, in large-scale image recognition, classic ResNetlike architectures are still state of the art (Mahajan et al., 2018; Xie et al., 2020; Kolesnikov et al.,
2020).
Inspired by the Transformer scaling successes in NLP, we experiment with applying a standard
Transformer directly to images, with the fewest possible modifications. To do so, we split an image
into patches and provide the sequence of linear embeddings of these patches as an input to a Transformer. Image patches are treated the same way as tokens (words) in an NLP application. We train
the model on image classification in supervised fashion.
When trained on mid-sized datasets such as ImageNet without strong regularization, these models yield modest accuracies of a few percentage points below ResNets of comparable size. This
seemingly discouraging outcome may be expected: Transformers lack some of the inductive biases
1
Fine-tuning code and pre-trained models are available at https://github.com/
google-research/vision_transformer
1
arXiv:2010.11929v2 [cs.CV] 3 Jun 2021
Published as a conference paper at ICLR 2021
inherent to CNNs, such as translation equivariance and locality, and therefore do not generalize well
when trained on insufficient amounts of data.
However, the picture changes if the models are trained on larger datasets (14M-300M images). We
find that large scale training trumps inductive bias. Our Vision Transformer (ViT) attains excellent
results when pre-trained at sufficient scale and transferred to tasks with fewer datapoints. When
pre-trained on the public ImageNet-21k dataset or the in-house JFT-300M dataset, ViT approaches
or beats state of the art on multiple image recognition benchmarks. In particular, the best model
reaches the accuracy of 88.55% on ImageNet, 90.72% on ImageNet-ReaL, 94.55% on CIFAR-100,
and 77.63% on the VTAB suite of 19 tasks.
2 RELATED WORK
Transformers were proposed by Vaswani et al. (2017) for machine translation, and have since become the state of the art method in many NLP tasks. Large Transformer-based models are often
pre-trained on large corpora and then fine-tuned for the task at hand: BERT (Devlin et al., 2019)
uses a denoising self-supervised pre-training task, while the GPT line of work uses language modeling as its pre-training task (Radford et al., 2018; 2019; Brown et al., 2020).
Naive application of self-attention to images would require that each pixel attends to every other
pixel. With quadratic cost in the number of pixels, this does not scale to realistic input sizes. Thus,
to apply Transformers in the context of image processing, several approximations have been tried in
the past. Parmar et al. (2018) applied the self-attention only in local neighborhoods for each query
pixel instead of globally. Such local multi-head dot-product self attention blocks can completely
replace convolutions (Hu et al., 2019; Ramachandran et al., 2019; Zhao et al., 2020). In a different
line of work, Sparse Transformers (Child et al., 2019) employ scalable approximations to global selfattention in order to be applicable to images. An alternative way to scale attention is to apply it in
blocks of varying sizes (Weissenborn et al., 2019), in the extreme case only along individual axes (Ho
et al., 2019; Wang et al., 2020a). Many of these specialized attention architectures demonstrate
promising results on computer vision tasks, but require complex engineering to be implemented
efficiently on hardware accelerators.
Most related to ours is the model of Cordonnier et al. (2020), which extracts patches of size 2 × 2
from the input image and applies full self-attention on top. This model is very similar to ViT,
but our work goes further to demonstrate that large scale pre-training makes vanilla transformers
competitive with (or even better than) state-of-the-art CNNs. Moreover, Cordonnier et al. (2020)
use a small patch size of 2 × 2 pixels, which makes the model applicable only to small-resolution
images, while we handle medium-resolution images as well.
There has also been a lot of interest in combining convolutional neural networks (CNNs) with forms
of self-attention, e.g. by augmenting feature maps for image classification (Bello et al., 2019) or by
further processing the output of a CNN using self-attention, e.g. for object detection (Hu et al., 2018;
Carion et al., 2020), video processing (Wang et al., 2018; Sun et al., 2019), image classification (Wu
et al., 2020), unsupervised object discovery (Locatello et al., 2020), or unified text-vision tasks (Chen
et al., 2020c; Lu et al., 2019; Li et al., 2019).
Another recent related model is image GPT (iGPT) (Chen et al., 2020a), which applies Transformers
to image pixels after reducing image resolution and color space. The model is trained in an unsupervised fashion as a generative model, and the resulting representation can then be fine-tuned or
probed linearly for classification performance, achieving a maximal accuracy of 72% on ImageNet.
Our work adds to the increasing collection of papers that explore image recognition at larger scales
than the standard ImageNet dataset. The use of additional data sources allows to achieve state-ofthe-art results on standard benchmarks (Mahajan et al., 2018; Touvron et al., 2019; Xie et al., 2020).
Moreover, Sun et al. (2017) study how CNN performance scales with dataset size, and Kolesnikov
et al. (2020); Djolonga et al. (2020) perform an empirical exploration of CNN transfer learning from
large scale datasets such as ImageNet-21k and JFT-300M. We focus on these two latter datasets as
well, but train Transformers instead of ResNet-based models used in prior works.
2
Published as a conference paper at ICLR 2021
Transformer Encoder
MLP
Head
Vision Transformer (ViT)
*
Linear Projection of Flattened Patches
* Extralearnable
[ cl ass] embedding
1 2 3 4 5 6 7 8 9 Patch + Position 0
Embedding
Class
Bird
Ball
Car
...
Embedded
Patches
Multi-Head
Attention
Norm
MLP
Norm
+
L x
+
Transformer Encoder
Figure 1: Model overview. We split an image into fixed-size patches, linearly embed each of them,
add position embeddings, and feed the resulting sequence of vectors to a standard Transformer
encoder. In order to perform classification, we use the standard approach of adding an extra learnable
“classification token” to the sequence. The illustration of the Transformer encoder was inspired by
Vaswani et al. (2017).
3 METHOD
In model design we follow the original Transformer (Vaswani et al., 2017) as closely as possible.
An advantage of this intentionally simple setup is that scalable NLP Transformer architectures – and
their efficient implementations – can be used almost out of the box.
3.1 VISION TRANSFORMER (VIT)
An overview of the model is depicted in Figure 1. The standard Transformer receives as input a 1D
sequence of token embeddings. To handle 2D images, we reshape the image x ∈ R
H×W×C into a
sequence of flattened 2D patches xp ∈ R
N×(P
2
·C)
, where (H, W) is the resolution of the original
image, C is the number of channels, (P, P) is the resolution of each image patch, and N = HW/P2
is the resulting number of patches, which also serves as the effective input sequence length for the
Transformer. The Transformer uses constant latent vector size D through all of its layers, so we
flatten the patches and map to D dimensions with a trainable linear projection (Eq. 1). We refer to
the output of this projection as the patch embeddings.
Similar to BERT’s [class] token, we prepend a learnable embedding to the sequence of embedded patches (z
0
0 = xclass), whose state at the output of the Transformer encoder (z
0
L
) serves as the
image representation y (Eq. 4). Both during pre-training and fine-tuning, a classification head is attached to z
0
L
. The classification head is implemented by a MLP with one hidden layer at pre-training
time and by a single linear layer at fine-tuning time.
Position embeddings are added to the patch embeddings to retain positional information. We use
standard learnable 1D position embeddings, since we have not observed significant performance
gains from using more advanced 2D-aware position embeddings (Appendix D.4). The resulting
sequence of embedding vectors serves as input to the encoder.
The Transformer encoder (Vaswani et al., 2017) consists of alternating layers of multiheaded selfattention (MSA, see Appendix A) and MLP blocks (Eq. 2, 3). Layernorm (LN) is applied before
every block, and residual connections after every block (Wang et al., 2019; Baevski & Auli, 2019).
3
Published as a conference paper at ICLR 2021
The MLP contains two layers with a GELU non-linearity.
z0 = [xclass; x
1
pE; x
2
pE; · · · ; x
N
p E] + Epos, E ∈ R
(P
2
·C)×D, Epos ∈ R
(N+1)×D (1)
z
0
` = MSA(LN(z`−1)) + z`−1, ` = 1 . . . L (2)
z` = MLP(LN(z
0
`)) + z
0
`, ` = 1 . . . L (3)
y = LN(z
0
L) (4)
Inductive bias. We note that Vision Transformer has much less image-specific inductive bias than
CNNs. In CNNs, locality, two-dimensional neighborhood structure, and translation equivariance are
baked into each layer throughout the whole model. In ViT, only MLP layers are local and translationally equivariant, while the self-attention layers are global. The two-dimensional neighborhood
structure is used very sparingly: in the beginning of the model by cutting the image into patches and
at fine-tuning time for adjusting the position embeddings for images of different resolution (as described below). Other than that, the position embeddings at initialization time carry no information
about the 2D positions of the patches and all spatial relations between the patches have to be learned
from scratch.
Hybrid Architecture. As an alternative to raw image patches, the input sequence can be formed
from feature maps of a CNN (LeCun et al., 1989). In this hybrid model, the patch embedding
projection E (Eq. 1) is applied to patches extracted from a CNN feature map. As a special case,
the patches can have spatial size 1x1, which means that the input sequence is obtained by simply
flattening the spatial dimensions of the feature map and projecting to the Transformer dimension.
The classification input embedding and position embeddings are added as described above.
3.2 FINE-TUNING AND HIGHER RESOLUTION
Typically, we pre-train ViT on large datasets, and fine-tune to (smaller) downstream tasks. For
this, we remove the pre-trained prediction head and attach a zero-initialized D × K feedforward
layer, where K is the number of downstream classes. It is often beneficial to fine-tune at higher
resolution than pre-training (Touvron et al., 2019; Kolesnikov et al., 2020). When feeding images
of higher resolution, we keep the patch size the same, which results in a larger effective sequence
length. The Vision Transformer can handle arbitrary sequence lengths (up to memory constraints),
however, the pre-trained position embeddings may no longer be meaningful. We therefore perform
2D interpolation of the pre-trained position embeddings, according to their location in the original
image. Note that this resolution adjustment and patch extraction are the only points at which an
inductive bias about the 2D structure of the images is manually injected into the Vision Transformer.
4 EXPERIMENTS
We evaluate the representation learning capabilities of ResNet, Vision Transformer (ViT), and the
hybrid. To understand the data requirements of each model, we pre-train on datasets of varying size
and evaluate many benchmark tasks. When considering the computational cost of pre-training the
model, ViT performs very favourably, attaining state of the art on most recognition benchmarks at
a lower pre-training cost. Lastly, we perform a small experiment using self-supervision, and show
that self-supervised ViT holds promise for the future.
4.1 SETUP
Datasets. To explore model scalability, we use the ILSVRC-2012 ImageNet dataset with 1k classes
and 1.3M images (we refer to it as ImageNet in what follows), its superset ImageNet-21k with
21k classes and 14M images (Deng et al., 2009), and JFT (Sun et al., 2017) with 18k classes and
303M high-resolution images. We de-duplicate the pre-training datasets w.r.t. the test sets of the
downstream tasks following Kolesnikov et al. (2020). We transfer the models trained on these
dataset to several benchmark tasks: ImageNet on the original validation labels and the cleaned-up
ReaL labels (Beyer et al., 2020), CIFAR-10/100 (Krizhevsky, 2009), Oxford-IIIT Pets (Parkhi et al.,
2012), and Oxford Flowers-102 (Nilsback & Zisserman, 2008). For these datasets, pre-processing
follows Kolesnikov et al. (2020).
4
Published as a conference paper at ICLR 2021
Model Layers Hidden size D MLP size Heads Params
ViT-Base 12 768 3072 12 86M
ViT-Large 24 1024 4096 16 307M
ViT-Huge 32 1280 5120 16 632M
Table 1: Details of Vision Transformer model variants.
We also evaluate on the 19-task VTAB classification suite (Zhai et al., 2019b). VTAB evaluates
low-data transfer to diverse tasks, using 1 000 training examples per task. The tasks are divided into
three groups: Natural – tasks like the above, Pets, CIFAR, etc. Specialized – medical and satellite
imagery, and Structured – tasks that require geometric understanding like localization.
Model Variants. We base ViT configurations on those used for BERT (Devlin et al., 2019), as
summarized in Table 1. The “Base” and “Large” models are directly adopted from BERT and we
add the larger “Huge” model. In what follows we use brief notation to indicate the model size and
the input patch size: for instance, ViT-L/16 means the “Large” variant with 16×16 input patch size.
Note that the Transformer’s sequence length is inversely proportional to the square of the patch size,
thus models with smaller patch size are computationally more expensive.
For the baseline CNNs, we use ResNet (He et al., 2016), but replace the Batch Normalization layers (Ioffe & Szegedy, 2015) with Group Normalization (Wu & He, 2018), and used standardized
convolutions (Qiao et al., 2019). These modifications improve transfer (Kolesnikov et al., 2020),
and we denote the modified model “ResNet (BiT)”. For the hybrids, we feed the intermediate feature maps into ViT with patch size of one “pixel”. To experiment with different sequence lengths,
we either (i) take the output of stage 4 of a regular ResNet50 or (ii) remove stage 4, place the same
number of layers in stage 3 (keeping the total number of layers), and take the output of this extended
stage 3. Option (ii) results in a 4x longer sequence length, and a more expensive ViT model.
Training & Fine-tuning. We train all models, including ResNets, using Adam (Kingma & Ba,
2015) with β1 = 0.9, β2 = 0.999, a batch size of 4096 and apply a high weight decay of 0.1, which
we found to be useful for transfer of all models (Appendix D.1 shows that, in contrast to common
practices, Adam works slightly better than SGD for ResNets in our setting). We use a linear learning
rate warmup and decay, see Appendix B.1 for details. For fine-tuning we use SGD with momentum,
batch size 512, for all models, see Appendix B.1.1. For ImageNet results in Table 2, we fine-tuned at
higher resolution: 512 for ViT-L/16 and 518 for ViT-H/14, and also used Polyak & Juditsky (1992)
averaging with a factor of 0.9999 (Ramachandran et al., 2019; Wang et al., 2020b).
Metrics. We report results on downstream datasets either through few-shot or fine-tuning accuracy.
Fine-tuning accuracies capture the performance of each model after fine-tuning it on the respective
dataset. Few-shot accuracies are obtained by solving a regularized least-squares regression problem
that maps the (frozen) representation of a subset of training images to {−1, 1}
K target vectors. This
formulation allows us to recover the exact solution in closed form. Though we mainly focus on
fine-tuning performance, we sometimes use linear few-shot accuracies for fast on-the-fly evaluation
where fine-tuning would be too costly.
4.2 COMPARISON TO STATE OF THE ART
We first compare our largest models – ViT-H/14 and ViT-L/16 – to state-of-the-art CNNs from
the literature. The first comparison point is Big Transfer (BiT) (Kolesnikov et al., 2020), which
performs supervised transfer learning with large ResNets. The second is Noisy Student (Xie et al.,
2020), which is a large EfficientNet trained using semi-supervised learning on ImageNet and JFT300M with the labels removed. Currently, Noisy Student is the state of the art on ImageNet and
BiT-L on the other datasets reported here. All models were trained on TPUv3 hardware, and we
report the number of TPUv3-core-days taken to pre-train each of them, that is, the number of TPU
v3 cores (2 per chip) used for training multiplied by the training time in days.
Table 2 shows the results. The smaller ViT-L/16 model pre-trained on JFT-300M outperforms BiT-L
(which is pre-trained on the same dataset) on all tasks, while requiring substantially less computational resources to train. The larger model, ViT-H/14, further improves the performance, especially
on the more challenging datasets – ImageNet, CIFAR-100, and the VTAB suite. Interestingly, this
5
Published as a conference paper at ICLR 2021
Ours-JFT Ours-JFT Ours-I21k BiT-L Noisy Student
(ViT-H/14) (ViT-L/16) (ViT-L/16) (ResNet152x4) (EfficientNet-L2)
ImageNet 88.55 ± 0.04 87.76 ± 0.03 85.30 ± 0.02 87.54 ± 0.02 88.4/88.5
∗
ImageNet ReaL 90.72 ± 0.05 90.54 ± 0.03 88.62 ± 0.05 90.54 90.55
CIFAR-10 99.50 ± 0.06 99.42 ± 0.03 99.15 ± 0.03 99.37 ± 0.06 −
CIFAR-100 94.55 ± 0.04 93.90 ± 0.05 93.25 ± 0.05 93.51 ± 0.08 −
Oxford-IIIT Pets 97.56 ± 0.03 97.32 ± 0.11 94.67 ± 0.15 96.62 ± 0.23 −
Oxford Flowers-102 99.68 ± 0.02 99.74 ± 0.00 99.61 ± 0.02 99.63 ± 0.03 −
VTAB (19 tasks) 77.63 ± 0.23 76.28 ± 0.46 72.72 ± 0.21 76.29 ± 1.70 −
TPUv3-core-days 2.5k 0.68k 0.23k 9.9k 12.3k
Table 2: Comparison with state of the art on popular image classification benchmarks. We report mean and standard deviation of the accuracies, averaged over three fine-tuning runs. Vision
Transformer models pre-trained on the JFT-300M dataset outperform ResNet-based baselines on all
datasets, while taking substantially less computational resources to pre-train. ViT pre-trained on the
smaller public ImageNet-21k dataset performs well too. ∗Slightly improved 88.5% result reported
in Touvron et al. (2020).
VTAB (19 tasks)
65
70
75
80
Accuracy [%]
Natural (7 tasks)
70
80
90
Specialized (4 tasks)
80
82
85
88
90
Structured (8 tasks)
50
60
70 ViT-H/14 BiT-L (R152x4) VIVI-Ex-100% (R50x3) S4L (R50x1)
Figure 2: Breakdown of VTAB performance in Natural, Specialized, and Structured task groups.
model still took substantially less compute to pre-train than prior state of the art. However, we note
that pre-training efficiency may be affected not only by the architecture choice, but also other parameters, such as training schedule, optimizer, weight decay, etc. We provide a controlled study of
performance vs. compute for different architectures in Section 4.4. Finally, the ViT-L/16 model
pre-trained on the public ImageNet-21k dataset performs well on most datasets too, while taking
fewer resources to pre-train: it could be trained using a standard cloud TPUv3 with 8 cores in approximately 30 days.
Figure 2 decomposes the VTAB tasks into their respective groups, and compares to previous SOTA
methods on this benchmark: BiT, VIVI – a ResNet co-trained on ImageNet and Youtube (Tschannen
et al., 2020), and S4L – supervised plus semi-supervised learning on ImageNet (Zhai et al., 2019a).
ViT-H/14 outperforms BiT-R152x4, and other methods, on the Natural and Structured tasks. On the
Specialized the performance of the top two models is similar.
4.3 PRE-TRAINING DATA REQUIREMENTS
The Vision Transformer performs well when pre-trained on a large JFT-300M dataset. With fewer
inductive biases for vision than ResNets, how crucial is the dataset size? We perform two series of
experiments.
First, we pre-train ViT models on datasets of increasing size: ImageNet, ImageNet-21k, and JFT300M. To boost the performance on the smaller datasets, we optimize three basic regularization
parameters – weight decay, dropout, and label smoothing. Figure 3 shows the results after finetuning to ImageNet (results on other datasets are shown in Table 5)2
. When pre-trained on the
smallest dataset, ImageNet, ViT-Large models underperform compared to ViT-Base models, despite
(moderate) regularization. With ImageNet-21k pre-training, their performances are similar. Only
with JFT-300M, do we see the full benefit of larger models. Figure 3 also shows the performance
2Note that the ImageNet pre-trained models are also fine-tuned, but again on ImageNet. This is because the
resolution increase during fine-tuning improves the performance.
6
Published as a conference paper at ICLR 2021
ImageNet ImageNet-21k JFT-300M
Pre-training dataset
70
75
80
85
90
ImageNet Top1 Accuracy [%]
BiT
ViT-B/32
ViT-B/16
ViT-L/32
ViT-L/16
ViT-H/14
Figure 3: Transfer to ImageNet. While
large ViT models perform worse than BiT
ResNets (shaded area) when pre-trained on
small datasets, they shine when pre-trained on
larger datasets. Similarly, larger ViT variants
overtake smaller ones as the dataset grows.
10 M 30 M 100 M 300 M
Number of JFT pre-training samples
30
40
50
60
70
Linear 5-shot ImageNet Top1 [%]
ViT-L/16
ViT-L/32
ViT-B/32
ViT-b/32
ResNet50x1 (BiT)
ResNet152x2 (BiT)
Figure 4: Linear few-shot evaluation on ImageNet versus pre-training size. ResNets perform better with smaller pre-training datasets
but plateau sooner than ViT, which performs
better with larger pre-training. ViT-b is ViT-B
with all hidden dimensions halved.
10
2 10
3
90
95
Transfer accuracy [%]
Average-5
Transformer (ViT)
ResNet (BiT)
Hybrid
10
2 10
3
75
80
85
90
ImageNet
Transformer (ViT)
ResNet (BiT)
Hybrid
Total pre-training compute [exaFLOPs]
Figure 5: Performance versus pre-training compute for different architectures: Vision Transformers,
ResNets, and hybrids. Vision Transformers generally outperform ResNets with the same computational budget. Hybrids improve upon pure Transformers for smaller model sizes, but the gap
vanishes for larger models.
region spanned by BiT models of different sizes. The BiT CNNs outperform ViT on ImageNet, but
with the larger datasets, ViT overtakes.
Second, we train our models on random subsets of 9M, 30M, and 90M as well as the full JFT300M dataset. We do not perform additional regularization on the smaller subsets and use the same
hyper-parameters for all settings. This way, we assess the intrinsic model properties, and not the
effect of regularization. We do, however, use early-stopping, and report the best validation accuracy
achieved during training. To save compute, we report few-shot linear accuracy instead of full finetuning accuracy. Figure 4 contains the results. Vision Transformers overfit more than ResNets with
comparable computational cost on smaller datasets. For example, ViT-B/32 is slightly faster than
ResNet50; it performs much worse on the 9M subset, but better on 90M+ subsets. The same is true
for ResNet152x2 and ViT-L/16. This result reinforces the intuition that the convolutional inductive
bias is useful for smaller datasets, but for larger ones, learning the relevant patterns directly from
data is sufficient, even beneficial.
Overall, the few-shot results on ImageNet (Figure 4), as well as the low-data results on VTAB
(Table 2) seem promising for very low-data transfer. Further analysis of few-shot properties of ViT
is an exciting direction of future work.
7
Published as a conference paper at ICLR 2021
4.4 SCALING STUDY
We perform a controlled scaling study of different models by evaluating transfer performance from
JFT-300M. In this setting data size does not bottleneck the models’ performances, and we assess
performance versus pre-training cost of each model. The model set includes: 7 ResNets, R50x1,
R50x2 R101x1, R152x1, R152x2, pre-trained for 7 epochs, plus R152x2 and R200x3 pre-trained
for 14 epochs; 6 Vision Transformers, ViT-B/32, B/16, L/32, L/16, pre-trained for 7 epochs, plus
L/16 and H/14 pre-trained for 14 epochs; and 5 hybrids, R50+ViT-B/32, B/16, L/32, L/16 pretrained for 7 epochs, plus R50+ViT-L/16 pre-trained for 14 epochs (for hybrids, the number at the
end of the model name stands not for the patch size, but for the total dowsampling ratio in the ResNet
backbone).
Figure 5 contains the transfer performance versus total pre-training compute (see Appendix D.5
for details on computational costs). Detailed results per model are provided in Table 6 in the Appendix. A few patterns can be observed. First, Vision Transformers dominate ResNets on the
performance/compute trade-off. ViT uses approximately 2 − 4× less compute to attain the same
performance (average over 5 datasets). Second, hybrids slightly outperform ViT at small computational budgets, but the difference vanishes for larger models. This result is somewhat surprising,
since one might expect convolutional local feature processing to assist ViT at any size. Third, Vision
Transformers appear not to saturate within the range tried, motivating future scaling efforts.
4.5 INSPECTING VISION TRANSFORMER
Input Attention
Figure 6: Representative examples of attention from the
output token to the input
space. See Appendix D.7 for
details.
To begin to understand how the Vision Transformer processes image data, we analyze its internal representations. The first layer of
the Vision Transformer linearly projects the flattened patches into a
lower-dimensional space (Eq. 1). Figure 7 (left) shows the top principal components of the the learned embedding filters. The components resemble plausible basis functions for a low-dimensional
representation of the fine structure within each patch.
After the projection, a learned position embedding is added to the
patch representations. Figure 7 (center) shows that the model learns
to encode distance within the image in the similarity of position embeddings, i.e. closer patches tend to have more similar position embeddings. Further, the row-column structure appears; patches in the
same row/column have similar embeddings. Finally, a sinusoidal
structure is sometimes apparent for larger grids (Appendix D). That
the position embeddings learn to represent 2D image topology explains why hand-crafted 2D-aware embedding variants do not yield
improvements (Appendix D.4).
Self-attention allows ViT to integrate information across the entire
image even in the lowest layers. We investigate to what degree
the network makes use of this capability. Specifically, we compute
the average distance in image space across which information is
integrated, based on the attention weights (Figure 7, right). This
“attention distance” is analogous to receptive field size in CNNs.
We find that some heads attend to most of the image already in the lowest layers, showing that
the ability to integrate information globally is indeed used by the model. Other attention heads
have consistently small attention distances in the low layers. This highly localized attention is
less pronounced in hybrid models that apply a ResNet before the Transformer (Figure 7, right),
suggesting that it may serve a similar function as early convolutional layers in CNNs. Further, the
attention distance increases with network depth. Globally, we find that the model attends to image
regions that are semantically relevant for classification (Figure 6).
4.6 SELF-SUPERVISION
Transformers show impressive performance on NLP tasks. However, much of their success stems
not only from their excellent scalability but also from large scale self-supervised pre-training (Devlin
8
Published as a conference paper at ICLR 2021
RGB embedding filters
(first 28 principal components)
1 2 3 4 5 6 7
Input patch column
1
2
3
4
5
6
7
Input patch row
Position embedding similarity
1
1
Cosine similarity
0 5 10 15 20
Network depth (layer)
0
20
40
60
80
100
120
Mean attention distance (pixels)
ViT-L/16
Head 1
Head 2
Head 3
...
Figure 7: Left: Filters of the initial linear embedding of RGB values of ViT-L/32. Center: Similarity of position embeddings of ViT-L/32. Tiles show the cosine similarity between the position
embedding of the patch with the indicated row and column and the position embeddings of all other
patches. Right: Size of attended area by head and network depth. Each dot shows the mean attention
distance across images for one of 16 heads at one layer. See Appendix D.7 for details.
et al., 2019; Radford et al., 2018). We also perform a preliminary exploration on masked patch
prediction for self-supervision, mimicking the masked language modeling task used in BERT. With
self-supervised pre-training, our smaller ViT-B/16 model achieves 79.9% accuracy on ImageNet, a
significant improvement of 2% to training from scratch, but still 4% behind supervised pre-training.
Appendix B.1.2 contains further details. We leave exploration of contrastive pre-training (Chen
et al., 2020b; He et al., 2020; Bachman et al., 2019; Henaff et al., 2020) to future work. ´
5 CONCLUSION
We have explored the direct application of Transformers to image recognition. Unlike prior works
using self-attention in computer vision, we do not introduce image-specific inductive biases into
the architecture apart from the initial patch extraction step. Instead, we interpret an image as a
sequence of patches and process it by a standard Transformer encoder as used in NLP. This simple,
yet scalable, strategy works surprisingly well when coupled with pre-training on large datasets.
Thus, Vision Transformer matches or exceeds the state of the art on many image classification
datasets, whilst being relatively cheap to pre-train.
While these initial results are encouraging, many challenges remain. One is to apply ViT to other
computer vision tasks, such as detection and segmentation. Our results, coupled with those in Carion
et al. (2020), indicate the promise of this approach. Another challenge is to continue exploring selfsupervised pre-training methods. Our initial experiments show improvement from self-supervised
pre-training, but there is still large gap between self-supervised and large-scale supervised pretraining. Finally, further scaling of ViT would likely lead to improved performance.