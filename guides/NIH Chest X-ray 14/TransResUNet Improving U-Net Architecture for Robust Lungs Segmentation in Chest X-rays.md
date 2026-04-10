Sakib Reza∗
, Ohida Binte Amin†
, and M.M.A. Hashem‡
Department of Computer Science and Engineering
Khulna University of Engineering & Technology (KUET)
Khulna 9203, Bangladesh
<sakibreza1@gmail.com>∗
, <ohida.amin2010@gmail.com>†
, <hashem@cse.kuet.ac.bd>‡
Abstract—Medical image segmentation is regarded as an important component in a computer-aided diagnosis (CAD) system
as it directly affects overall system performance. In this paper,
we propose a new fully convolutional encoder-decoder model
for lung segmentation named TransResUNet. We developed this
architecture improving the state-of-the-art U-Net model. As part
of the improvement to the classical U-Net, we introduced a pretrained encoder, a special skip connection and a post-processing
module in the proposed architecture. As a result, the proposed
model outperformed the baseline U-Net model by 97.6% vs
94.9% considering the dice coefficient and 98.5% vs. 96.8% in
terms of accuracy. The proposed TransResUNet achieved this
feat with about 24% fewer parameters than the baseline U-Net.
The implementation (based on Keras) of our proposed model
is publicly available at <https://sakibreza.github.io/TransResUNet>
with additional resources.
Index Terms—Chest X-ray; Deep Learning; Lungs; Segmentation.
I. INTRODUCTION
The chest X-ray (CXR) is considered a common way to
diagnose a variety of diseases related to lungs and heart.
One of the major applications of CXR is to assist diagnosis
and inspect treatment for different lung conditions such as
pneumonia, emphysema, and cancer. Concerning these sorts of
diseases, a radiologist generally focuses on the lungs region
for an effective interpretation of CXRs [1]. In the case of
developing a computer-aided diagnosis (CAD) system, the
interpretation concepts also remain the same since a diagnosis
system should mimic a human radiologist. Therefore, in the
development pipeline of such a CAD system, we need to
include an effective lung segmentation unit.
In recent years, impressive progress has been made in the
domain of deep learning and computer vision. A range of
medical image segmentation models has also been developed
with this advancement. U-Net [2] architecture is one of them
and is regarded as a standard model for medical image
segmentation. Regardless of its wide acceptability, we still
found some scopes of alterations to make it more improved. In
this paper, we present a new fully-convolutional model for lung
segmentation called TransResUNet transforming the classical
U-Net architecture.
The composition of this paper is as follows. Section II
and Section III describe the baseline U-Net model and the
proposed model respectively. The experimental results are
discussed in Section IV with a brief description of the data.
Finally, Section V concludes the paper.
II. BASELINE U-NET MODEL
The architecture of U-Net [2] model consists of two modules - encoder and decoder. The role of the encoder is to
extract spatial features of input images and the decoder part
builds the mapping of segmentation using encoded features.
The encoder module is quite similar to regular convolutional
network architecture. Here, an input image goes through a
number of blocks consisting of two padded 3 × 3 convolution
layers with ReLU activation and max-pooling consecutively.
After each max-pooling operation, the number of feature
channels gets double and spatial dimensions become half of
its previous state. Conversely, the decoder module up-samples
the map of features, concatenates it with the corresponding
decoder feature map and then conducts two 3 × 3 convolution
operations with ReLU activation. In each up-sampling step,
the decoder performs halving the number of feature channels
and simultaneously their spatial dimension gets double. Thus,
the original dimension of the input image is preserved and the
network outputs the expected segmentation mask.
III. PROPOSED TRANSRESUNET MODEL
Focusing on the lung segmentation, we made some modifications in the classical U-Net model and proposed a new
model named TransResUNet. The proposed model is demonstrated in figure 1 with all required definitions. The major
components of our proposed method are described as follows.

1) Pre-trained Encoder: Employing a pre-trained encoder
can improve the training of an encoder-decoder network [3].
Therefore, as an encoder, we introduced a trimmed portion of
the VGG-16 model [4] pre-trained on the ImageNet dataset
[5]. In a transfer learning task, the lower level weights have
a greater impact as they are focused more on the primitive
feature extraction. The weights of higher layers are rather
dedicated to specialized feature extraction for the target task.
Considering these facts, we took only the first 7 layers of
VGG-16 keeping the architecture as compact as possible.
Here, first, the input grayscale image is converted to a 3
channel image to make it compatible with the pre-trained
encoder and then it enters the main encoder. The first two
encoder blocks contain two sets of 3X3 convolutions with
978-1-7281-7366-5/20/$31.00 ©2020 IEEE
Fig. 1: Proposed TransResUNet model for lungs segmentation
ReLU activation and a max-pooling layer. The only difference
in the last encoder block is that it holds three sets of 3X3
convolutions instead of two.
2) Modified Skip Connection: The U-Net architecture introduced us with an inventive approach to propagate spatial
information from encoder to decoder which is known as skip
connections. These connections generally join the encoder
layers prior to max-pooling and the corresponding decoder
layers after up-sampling. In spite of conserving the diffused
spatial features, the traditional skip connection may cause a
semantic gap that appears between the concatenated encoder
and decoder features. For example, the first skip connection
joins the encoder prior to the first max-pooling layer with
the decoder following the last up-sampling process. In this
context, the features originating from the encoder module
are assumed to be comparatively primitive features since
they proceed through less computational processing. On the
other hand, features produced form the decoder module are
assumed to be more advanced-type as they go through a lot
of processing. To alleviate this semantic gap, it is supposed
to be a competent idea to use a series of convolution blocks
in the skip connection. Res path [6] is one of these types of
connection which can solve this situation. Figure 2 shows how
we constructed a res path using a number of residual blocks.
3) Decoder: The decoder module comprises three decoder
blocks. Each of these blocks consists of an up-sampling that
follows two 3X3 convolution layers with ReLU activation.
Here, we used the nearest neighbor up-sampling as it is
effective in limiting artifacts at the edge of the predicted
output form. Again, features of the last convolution layer
in a decoder block are concatenated with the corresponding
encoded features from its residual connection. Finally, the
last layer of the decoder produces the predicted output mask
followed by a 1X1 convolution and sigmoid activation.
4) Post-processing: While experimenting with our model,
we encountered that a few predicted masks have defects likeholes in the mask, overlapping left and right masks, and
unexpected objects out of the real mask region. To reduce
these output noises, we added a post-processing segment at
Fig. 2: Res path contains blocks of two parallel 3X3 and 1X1
convolution filter and the addition of their outputs. In proposed
model, the first, second and third res path (according to figure
5) contain 3, 2 and 1 such block(s) respectively.
the end of our lung segmentation pipeline. The proposed postprocessing steps (figure 3) are performed as follows.
First, we applied the flood-fill algorithm for filling up the
undesired holes in the mask. The algorithm is performed by
choosing the top left corner pixel (0,0) of the mask as the
starting node. Here, the target color was black(0) like the
starting node and the replacement color was white(1). The
resultant image was inverted and unioned (bitwise OR) with
the main image to get the desired output similar to figure 3(a).
Fig. 3: Visualization of the post-processing steps
Secondly, there can remain some undesired objects around
the lung masks. To remove them, we determined all the
connected components in the mask and eliminated all the
components with an area less than a predefined threshold
value. Finally, we performed a morphological opening on the
predicated mask using an elliptical structuring element. Figure
3(c) visualizes how opening detaches the left and right masks
on an overlapped mask sample.
IV. EXPERIMENTAL RESULTS
A. Data
For our experimental purpose, we used the Montgomery
Country (MC) dataset [7]. This dataset is managed by the
National Library of Medicine (NLM) in collaboration with the
Department of Health and Human Services of Montgomery
Country, Maryland. It consists of 138 posterior-anterior chest
X-Ray images including 80 normal and 58 abnormal X-Rays.
These images are provided with corresponding manual lung
masks that were annotated with the support of professional
radiologists. For our purpose, we resized all the images to the
pixel resolution of 512 × 512.
B. Evaluation Metrics
In model evaluation, we applied two widely used metrics
for image segmentation tasks - dice similarity coefficient and
pixel accuracy. The definitions of these metrics are as follows.
a) Dice Similarity Coeffecient: The dice coefficient is a
standard metric for evaluating segmentation performance. It is
defined as twice of the overlapped area between the predicted
mask and the ground truth mask divided by the total area sum
of the predicted mask and the ground truth mask. Equation 1
shows how this metric is calculated.
Dice =
2 |PM ∩ GM|
|PM| + |GM|
=
2 |T P|
2 |T P| + |F N| + |F P|
(1)
Here, PM = predicted mask and GM = Ground Truth
Mask.
b) Pixel Accuracy: The pixel accuracy is the simplest
way to measure segmentation performance. Pixel accuracy is
defined as the sum of the number of true positive and true
negative pixels divided by the total number of pixels. Equation
2 presents the process of calculating the pixel accuracy.
Accuracy =
|T P| + |T N|
|T P| + |T N| + |F P| + |F N|
(2)
Here, TP = True Positives, TN = True Negatives, FP = False
Positives and FN = False Negatives.
C. Model Evaluation
For evaluation of our proposed method, we randomly split
our dataset into three segments - train set (70%), validation
set(10%), and test set (20%). Then, we trained the baseline
U-Net model and the proposed TransResUNet model for 150
epochs on a Tesla K80 GPU with data augmentation.
Fig. 4: Progress comparison of validation performance with
the number of training epochs.
In the training phase, we observed the validation progress
of both the baseline and the proposed model. Fig 4 shows the
comparison of validation progress between two models. From
this, we notice that the proposed model with a pre-trained
encoder reached a fair state with a minimal number of epochs.
Eventually, it outperforms the U-Net model in both metrics -
dice similarity coefficient and accuracy (binary accuracy).
In addition to comparing the proposed model with U-Net,
we set up two variants of the proposed model - one without
pre-trained weight and another without res path for further justification. Table I demonstrates test results comparison among
the baseline U-Net, variants of TransResUNet and finally the
proposed TransResUNet model. Here, we see that the proposed
TransResUNet outperformed the baseline U-Net model with
24.4% fewer parameters. Furthermore, the proposed model is
proved better than its variants which validates the addition of
pre-trained encoder and res path. For a simpler interpretation
of the result, we visualize some sample predictions of our
model comparing with the predictions of baseline U-Net and
corresponding ground truths in figure 5.
TABLE I
SEGMENTATION RESULTS COMPARISON AMONG THE
BASELINE U-NET MODEL, VARIANTS OF TRANSRESUNET
AND THE PROPOSED TRANSRESUNET MODEL
Model Param.* Dice (%) Acc. (%)
U-Net (baseline) 7.8M 94.9 96.8
TransResUNet (random initialization) 5.9M 92.1 95.3
TransResUNet (without res. path) 5.8M 94.0 96.9
TransResUNet (proposed) 5.9M 97.6 98.5
∗Parameters in Millions (M).
Fig. 5: Comparison of sample segmentation predictions. Here,
masks are shown in the form of color maps over input images.
Finally, Table II shows the comparison among different
studies conducted on the same dataset. From other studies,
S. Candemir et. al. [7] proposed an approach which involves
anatomical atlases with non-rigid registration. Apart from this,
P. Chondro et. al. [8] introduced a low order region growing
method to segment the lungs. The most recognized study
on the lungs segmentation task is conducted by et. al. [9].
They used state-of-the-art U-Net architecture which is the
baseline model in our study. This comparison further verifies
the robustness of our model on lung segmentation.
TABLE II
COMPARISON OF LUNGS SEGMENTATION STUDIES ON MC
DATASET
Approach Dice (%) Acc. (%)
Anatomical atlases with Non-rigid Registration [7] - 94.1
Low order adaptive region growing [8] - 96.6
U-Net [9] 95.4 97.7
TransResUNet (proposed) 97.6 98.5
V. CONCLUSION
The results show that our segmentation model improved the
effectivity of classical U-Net with less number of parameters.
One of the major limitations of using such fully convolutional
model is that they need a large number of training data. In
our model, we have solved this problem by using a transfer
learning approach. Furthermore, we alleviated a semantic gap
issue in the skip connection by replacing it with a residual
connection. As a future plan, we would like to evaluate
the TransResUNet model with other medical image datasets,
modify it further and transform it into a generic model for
medical image segmentation.
