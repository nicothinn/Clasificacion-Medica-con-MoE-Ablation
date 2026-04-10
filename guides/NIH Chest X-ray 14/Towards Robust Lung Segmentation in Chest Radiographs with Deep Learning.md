ent of Computer Science
Georgia State University
<jislam2@student.gsu.edu>
Yanqing Zhang
Department of Computer Science
Georgia State University
<yzhang@gsu.edu>
Abstract
Automated segmentation of Lungs plays a crucial role in the computer-aided diagnosis of chest X-Ray (CXR) images. Developing an efficient Lung segmentation
model is challenging because of difficulties such as the presence of several edges
at the rib cage and clavicle, inconsistent lung shape among different individuals,
and the appearance of the lung apex. In this paper, we propose a robust model for
Lung segmentation in Chest Radiographs. Our model learns to ignore the irrelevant
regions in an input Chest Radiograph while highlighting regions useful for lung
segmentation. The proposed model is evaluated on two public chest X-Ray datasets
(Montgomery County, MD, USA, and Shenzhen No. 3 People’s Hospital in China).
The experimental result with a DICE score of 98.6% demonstrates the robustness
of our proposed lung segmentation approach.
1 Introduction
Chest Radiographs are the most common radiological procedure and constitute about one-third of all
radiological procedures [1]. Chest X-Rays (CXR) are used to study various structures such as the
heart and lungs for several disease diagnosis including Lung cancer, Tuberculosis, Pneumonia etc.
For computer-aided diagnostic systems, segmentation of anatomical structures in chest X-Rays plays
an important role. For example, irregular shape, size measurements and total lung area can provide
significant insight about early manifestations of life-threating diseases, including cardiomegaly,
emphysema etc. The performance of Lung segmentation plays a vital role in such applications.
Accurate Lung segmentation is considered as one of the challenges in medical image analysis due
to the shape variance caused by age, gender and health. If there are a presence of external objects,
such as cardiac pacemakers, surgical clips, and sternal wire, automated segmentation of Lung fields
becomes more difficult.
There are four major categories of Lung segmentation methods. Rule-based models use predefined
anatomical rules for lung segmentation [2], [3]. Pixel-based methods try to label each pixel as lung or
non-lung [4]. Deformable-based models use object shape and image appearance [5]. Registration
based models match and refine lung fields based on a segmented lung database [6]. Most of the
traditional models use hand-crafted shape and region information for Lung segmentation. In recent
days, the advancement of deep learning technologies [7], [8], [9], [10], [11] is transforming the
medical image analysis world with great success . Now we can develop robust frameworks to learn
useful features directly from the input data for the segmentation task.
For our current work, we develop an automated framework for Lung segmentation in chest X-Ray
images using a Deep Convolutional Neural Network based on the U-Net [12]. Besides, with several
experiments, we demonstrate that proper data augmentation and network architecture can significantly
improve the performance for lung segmentation in chest X-Ray images.
Machine Learning for Health (ML4H) Workshop at NeurIPS 2018.
arXiv:1811.12638v1 [cs.CV] 30 Nov 2018
(a) (b) (c)
Figure 1: Sample data from Montgomery Dataset: (a) Chest X-Ray (b) Left mask; (c) Right mask
(a) (b)
Figure 2: Sample data from Shenzhen Dataset: (a) Chest X-Ray (b) mask
2 Method
2.1 Data
We used two datasets that include publicly available datasets from Montgomery County, Maryland,
and Shenzhen No. 3 People’s Hospital in China. These datasets are maintained by the National
Library of Medicine (NLM), National Institutes of Health (NIH) [6]. In the Montgomery County
X-Ray Set, there are 138 posterior-anterior X-Rays (80 X-Rays are normal, and 58 X-Rays have a
wide range of abnormalities, including effusions and miliary patterns). Shenzhen Hospital X-Ray Set
have 340 normal X-Rays and 275 abnormal X-Rays showing various manifestations of tuberculosis.
Figure 1 shows sample data from the Montgomery dataset and Figure 2 shows sample data from the
Shenzhen dataset.
2.2 Proposed Network Architecture
Our proposed Lung segmentation model is based on the U-Net [12] architecture. The proposed model
is shown at Figure 3. Several data augmentation techniques such as zooming, cropping, horizontal
flipping etc. are performed on the input dataset for increasing the training data volume. After data
pre-processing and data augmentation, each image is resized to 512*512 dimension. Similar to
U-Net [12], the lung segmentation network consists of a contracting path and an expansive path.
Upsampling of the feature map in the expansive path is combined with the high resolution features
from the contracting path to retain the segmentation information. Detail of the CNN architecture used
in the proposed lung segmentation model is shown in Figure 4.
Figure 3: Proposed Lung segmentation model.
2
Figure 4: Detail of the CNN architecture used in the proposed lung segmentation model.
3 Experiments
Each chest X-ray image was resized to 512*512 before passing to the Deep CNN model for segmentation. We have combined the left and right masks for each chest X-Ray of the Montgomery dataset
and performed Morphological transformations (Dilation) on the combined mask. We have developed
the proposed model using Keras framework and trained on NVIDIA GTX TITAN V GPU. We have
combined all the chest X-Rays from both Montgomery and Shenzhen dataset for building our input
dataset. The training set consists of 80% data of the total input dataset, and 20% data were used as
the test dataset. From the training dataset, 10% data were used as validation dataset. We trained
the proposed model using the Adam optimizer with learning rate of 0.0005, batch size of 4 and 200
epochs. For data augmentation, zoom range was set to 0.05, height shift, width shift and horizontal
shift was used.
3.1 Results
Table 1 shows the result of our proposed Lung segmentation model. We report the result in terms of
DICE coefficient [15] following previous research works. DICE coefficient is the overlap between
the ground truth, GT and the calculated segmented mask, S:
DICE =
|S ∩ GT|
|S| + |GT|
=

2|T P|
2|T P| + |F N| + |F P|
(1)
A model with a higher DICE score indicates better segmentation performance of the network. From
the result, we can see that data augmentation improves the performance of the proposed network by
around 2.2%. We have developed another CNN model using skip connections in the convolutional
layers. But skip connections/resnet blocks did not help to improve the segmentation result, hence
we are not reporting it here. Figure 5 shows the lung segmentation result of our proposed model.
The performance is consistent across different runs. From the figure, we can see that the predicted
segmentation of our proposed model matches very well with the manual segmentation ground truths.
3
Table 1: Performance comparison of the proposed model with previous state-of-the-arts.
Method Dice Coefficient
Candemir et al. [6] 94.1
ED-CNN [14] 97.4
FCN [13] 97.7
Proposed model 98.6
Figure 5: Sample segmentation result as compared with the manual ground truth segmentation.
4 Conclusion
The proposed model demonstrates robust performance for Lung segmentation from Chest X-Rays.
In future, we will evaluate the performance of our proposed model for other Chest X-Ray database
including JSRT. Additionally, instead of using another mask database, we will incorporate attention
mechanism in the proposed architecture for improving the segmentation result and performing weakly
supervised segmentation
