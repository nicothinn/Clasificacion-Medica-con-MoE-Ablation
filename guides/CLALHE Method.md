SECTION V.Proposed CLAHE Based on Local and Adaptive Concept Methodology
The proposed method, contrast limited adaptive local histogram equalization (CLALHE), is a new approach for the local enhancement of poor-contrast images based on an adaptive concept. Given the varying areas of contrast in each image type, CLALHE hypothesises that their enhancement weights must be suitably tuned for each area. This hypothesis was proposed because the foreground area contains essential objects, followed by the region of interest (ROI), which holds worthy image information. Meanwhile, the background contains less worthy and crucial information, such as the sky, the sea, and the wall and the cleared or black background areas in the medical field. The proposed method not only aims to improve the richness of information and details within an image but also preserve its overall structure.

The proposed CLALHE method varies from the conventional CLAHE in terms of the following criteria: (i) an adaptively and locally based concept for parameter computation based on input image information; (ii) selection process for optimal parameters; (iii) subdivision process of the input image to various subimages and separate processing of each.

This section discusses the theories related to the proposed CLALHE. In accordance with the criteria, the algorithm improves the overall image contrast and enhances local details to produce a visually pleasing image and preserve image structure. Figure 4 illustrates a graphical representation of the proposed method, with more information provided in subsequent subsections. The proposed CLALHE method comprises two parts, namely, (i) Part 1: determining optimum parameters and (ii) Part 2: obtaining a resultant image. As the proposed CLALHE concept is developed to be a local and adaptive enhancement technique. The first part of the proposed CLALHE method focuses on the computation of the image histogram to extract its peaks and valleys, which are further utilised to calculate other parameters. Furthermore, this part aims to identify the optimum parameters for Part 2 criteria. The second part of the proposed CLALHE method is an essential part, where the input image is subdivided into subimages and hence improved individually, beginning with the first subimage and ending with the last one based on the optimum parameters provided previously in Part 1. The proposed CLALHE differs from the conventional CLAHE in terms of the following:

In Part 1, three parameters are introduced: (i) context block dimension (CBD), (ii) N-parameter and (iii) ideal CL (I_CL). CBD refers to the optimum dimensions of the context regions for CLAHE (i.e. width and height). The N-parameter is introduced as a new parameter, and it plays a crucial role in calculating a specific value used in the I_CL parameter’s formula, which ensures that the I_CL remains within acceptable limits. An overly high I_CL can result in the overenhancement of the resultant image, which must be avoided. Lastly, the I_CL parameter is introduced as an equation to calculate the CL used in the proposed method.

In Part 2, two parameters are proposed: (i) optimum number of subimages (ONS) and (ii) subimage dimensions (SID). The ONS parameter is designed to obtain an even number, which is used to identify the best number of subimages to be used in subdividing the input image equally.

FIGURE 4. - Block diagram of the proposed CLALHE.

FIGURE 4.
Block diagram of the proposed CLALHE.

Show All

Moreover, the ONS is determined based on the number of valleys in the input image’s histogram. The SID is designed to calculate the dimensions of subimages based on the ONS, and it is used to subdivide the input image to subimages. These proposed parameters in Parts 1 and 2 guarantee the local and adaptive enhancement of input images to produce the best enhanced final image without washed-out or unwanted artifact details. Figure 5 depicts the operation flowchart of the proposed CLALHE method.

FIGURE 5. - Flowchart of the proposed CLALHE.

FIGURE 5.
Flowchart of the proposed CLALHE.

Show All

A. Part 1: Determining Optimum Parameters
This section discusses Part 1 of the proposed CLALHE technique in detail. Part 1 includes six crucial steps: (i) reading of the input image; (ii) acquisition of the image histogram; (iii) identification and quantification of the number of histogram peaks and valleys; (iv) computation of parameters (N-parameter, CBD, and I_CL); (v) application of CLAHE; (vi) sorting of the optimum enhancement parameters. Once an input image is provided, it will be converted into a greyscale scheme (Figure 6). By converting the RGB image to greyscale, we can apply the CLALHE method efficiently (in terms of processing time) because we only need to apply the technique to a single channel, which represents the image’s intensity (greyscale). The greyscale image retains the spatial information of the RGB image but without the colour information, which allows us to focus on enhancing the image contrast without dealing with multiple colour channels.

FIGURE 6. - Input image of the (a) original and (b) greyscale images.

FIGURE 6.
Input image of the (a) original and (b) greyscale images.

Show All

Figure 6 demonstrates the input image, and Figure 7 illustrates the histogram distribution of the input image. The next step is the identification of the number of peaks and valleys of the histogram. To identify a histogram peak, we compared the frequency (i.e. the pixel number) of each grey level in the histogram to those of its neighbouring grey levels on the right and left sides. If the frequency of a particular grey level was higher than that of its neighbours, then the pixel was identified as a peak. The number of peaks was then determined and denoted as NPeaks . Furthermore, for the identification of histogram valley, the frequency (i.e. the number of pixels) of each grey level in the image histogram was compared with those of its neighbouring grey levels on the right and left sides. If the frequency of a particular grey level was lower than that of its neighbours, then the pixel was considered a valley. The number of valleys is then quantified and denoted as Nvalleys .

FIGURE 7. - Histogram distribution of the greyscale image in Figure 6 (b).

FIGURE 7.
Histogram distribution of the greyscale image in Figure 6 (b).

Show All

To illustrate the identification process of histogram peaks and valleys, we selected a range of grey levels (i.e. 0 to 20) from the histogram distribution of the grey image in Figure 7. In accordance with the definition of identifying histogram peaks and valleys, we illustrated the detected peaks and valleys (Figure 8).

FIGURE 8. - Identified peaks and valleys for the selected sample (i.e. 0–20 grey levels). Note: The red triangle refers to the valley, and the green triangle indicates the peak.

FIGURE 8.
Identified peaks and valleys for the selected sample (i.e. 0–20 grey levels). Note: The red triangle refers to the valley, and the green triangle indicates the peak.

Show All

This illustration identified 2 peaks and 3 valleys. Table 3 tabulates the identified peaks and valleys, with grey levels of 7 and 15 for peaks and 2, 14 and 16 for valleys.

TABLE 3 Identified Peaks and Valleys in Figure 8. (0–20 Grey Levels)
Table 3- Identified Peaks and Valleys in Figure 8. (0–20 Grey Levels)

After the identification and quantification of the peaks and valleys numbers, their values were used as bases for the calculations of the newly introduced parameters (i.e. N-parameter, CBD and I_CL) in the next step. The N-parameter was calculated adaptively using the following equation:
N−parameter=∑Valley[i]NValleys(1)
View SourceRight-click on figure for MathML and additional features.where Valley[i] represents the frequency (i.e. pixel number) of each identified grey level as a valley in the image histogram, and NValleys refers to the number of identified valleys in the image histogram. Adaptive calculation of the CBD parameter is achieved using the following:
CBD={Height=⌈Ln(NPeaks)⌉Width=⌈Ln(NPeaks)⌉(2)
View SourceRight-click on figure for MathML and additional features.where in the expression ⌈Ln(NPeaks)⌉ , ⌈⌉ indicates the ceiling function, and it is used to round the natural logarithm of the variable NPeaks to the nearest whole number. If the natural logarithm of NPeaks , denoted as Ln(NPeaks) , results in a noninteger, then it is adjusted upward to the closest integer. This method ensures that the output values are integers with a nonlinear growth. Let us consider a case where the number of NPeaks is 52. Using Eq. (2), CBD will have the following dimensions:
HeightWidth=⌈Ln(52)⌉=⌈3.951⌉=4=⌈Ln(52)⌉=⌈3.951⌉=4
View SourceRight-click on figure for MathML and additional features.The value of 3.951 was rounded up to 4, and the CBD dimensions was set to 4×4 .

The I_CL parameter was calculated adaptively using the following equation:
I_CL=(Peaks[i]+N−parameter)Max(Valley)(3)
View SourceRight-click on figure for MathML and additional features.where Peaks[i] represents the frequency (i.e. pixel number) corresponding to each grey level identified as a peak in the image histogram. The term Max(Valley[i]) indicates the highest valley value within the group of identified valleys.

Once all required parameters for the application of the current procedure become available, the input image is subjected to the enhancement process multiple times. The frequency of enhancement process depends on the number of identified peaks (i.e. NPeaks ) in the image histogram. Therefore, this part tests each value (i.e. frequency) of the identified peaks group to specify the peak that generates the optimum enhanced image. To ensure that Part 1 will produce an optimum enhanced image, we proposed a new fitness function for this objective called the comprehensive image quality index (CIQI):
CIQI=(PSNR+Entropy)AMBE(4)
View SourceRight-click on figure for MathML and additional features.

According to Eq. (4), the CIQI function is contrived to highlight and enhance the enrichment of subimage information and local details without structural distortion. Entropy highlights and enhances the richness of local details and information, whereas the PSNR highlights noise reduction in the image. In addition, AMBE is used to highlight brightness consistency. The proposed CLALHE requires high PSNR and entropy and the lowest AMBE value to obtain the highest or optimal CIQI. High PSNR and entropy ensure noise reduction, and richness of information is enhanced and highlighted. A low AMBE value ensures that the enhancement process does not excessively alter the image’s natural appearance. The CIQI provides the following advantages:

Balancing of Image Quality Metrics:

PSNR is widely used in the assessment of the fidelity of image processing techniques, particularly in terms of noise reduction. High PSNR values indicate the proximity of the enhanced image to the original, unprocessed image, which contributes to the maintenance of image integrity.

Entropy estimates the quantity of information in an image, with high values denoting rich detail and desirable contrast. This condition ensures the attainment of an informative and visually appealing enhanced image.

AMBE refers to a measure of differences in the brightness between input and enhanced images. A low AMBE indicates that enhancement preserved the natural brightness of an image, which is crucial for maintaining visual consistency.

Complementary Evaluation:

By summing the values of PSNR and entropy, the equation recognises the importance of noise reduction and information content in the enhanced image and ensures that it is clear and detailed.

The division by AMBE introduces a penalty for excessive brightness distortion, which ensures that the enhancement process does not excessively alter an image’s natural appearance. This balance is crucial for applications requiring visual fidelity and information preservation.

In conclusion, the proposed CIQI equation offers a balanced and comprehensive metric for performance estimation of the proposed CLALHE method (Part 1). The equation helps in evaluating the proposed CLALHE in an informed and nuanced manner through the combination of noise reduction, detail preservation and brightness consistency into a single metric.
Lastly, the specific parameters of the optimum enhanced image (i.e. CBD and I_CL) are stored for further processing in Part 2.

B. Part 2: Obtaining A Resultant Image
Part 2 represents the final section of the proposed CLALHE method, which involves five crucial steps: (i) calculation of ONS, (ii) calculation of SID, (iii) subdivision of the input image to multiple subimages, (iv) utilisation of the optimum enhancement parameters obtained previously in Part 1 for the enhancement of each subimage; (v) combination of the enhanced subimages to produce the final enhanced image.

The first step involves the calculation of the ONS based on the NValleys represented in the following equation:
ONS≈{⌈log2(NValleys)⌉if ONS is an even number⌈log2(NValleys))⌉+1 if ONS is an odd number(5)
View SourceRight-click on figure for MathML and additional features.

From Eq. (5), the ONS is considered even and odd integer numbers. Therefore, if the numbers of histogram valleys (NValleys ) are 50 and 100, respectively, the ⌈log2(NValleys)⌉ produces values of 5.65 and 6.65, respectively. Thus, the ONS values were set to 6 (i.e. ⌈5.65⌉=6 ) and 8 (i.e., ⌈6.65⌉+1=8 ). The ONS obtained from Eq. (5) was then used to adaptively subdivide the input image into subimages. In this study, ONS was set to be an even number only. Thus, when Eq. (5) produced an odd number, 1 was added to convert the answer into an even number. Figures 9 and 10 demonstrate two examples of subdivision process for NS =2 and NS =4, respectively. Let us consider an image with dimensions M×N (i.e. M and N denote the numbers of columns and rows, respectively) subdivided into ONS subimages. After implementation of the procedure, the dimensions of each subimage (SID) were calculated as follows:
SID(Height,Width)=⎧⎩⎨⎪⎪⎪⎪⎪⎪(⌈M⌉,⌈NNS⌉) if NS=2(⌈M2⌉,⌈NNS⌉×2) if NS>2(6)
View SourceRight-click on figure for MathML and additional features.

FIGURE 9. - Image division with ONS =2 (a) right and (b) left subimages.

FIGURE 9.
Image division with ONS =2 (a) right and (b) left subimages.

Show All

FIGURE 10. - Image division with ONS =4 (a) first, (b) second, (c) third and (d) fourth subimages.

FIGURE 10.
Image division with ONS =4 (a) first, (b) second, (c) third and (d) fourth subimages.

Show All

For the image shown in Figure 9, each subimage has the following dimensions when using Eq. (5) (with original dimensions of 592 ×844 ):
Height=592Width=8962=448
View SourceRight-click on figure for MathML and additional features.Thus, the SID for each subimage is (592, 448).

On the other hand, for the image shown in Figure 10, each subimage has the following dimensions:
Height=5922=296Width=8964×2=448
View SourceRight-click on figure for MathML and additional features.Thus, the SIS for each subimage is (296, 448).

The next step includes the separate enhancement of each subimage using the optimum parameters (i.e. CBD and I_CL) identified in Part 1. Once all the subimages have been enhanced, they are then combined to construct the resultant image.

This approach ensures that the most suitable enhancement parameters are applied to each sub-image. The resulting image is a composite of enhanced subimages and offers an improved overall image quality. Figure 11 illustrates the pseudocode of the proposed CLALHE.

FIGURE 11. - Pseudocode of the proposed CLALHE method.

FIGURE 11.
Pseudocode of the proposed CLALHE method.

Show All

SECTION VI.Contribution of the Proposed CLALHE Method
The proposed CLALHE method theorises that dividing an image into subimages will help expand the intensity distribution range while processing the local features of each subimage. Unlike the conventional CLAHE method, the proposed CLALHE introduces a locally and adaptively based approach, sorting the optimal enhancement parameters to ensure that each subimage is efficiently enhanced before being combined to create the final image. Additionally, the approach adaptively computes these parameters, as outlined in Part 1: Determining Optimum Parameters. The proposed CLALHE overcomes and mitigates several disadvantages inherent in the conventional CLAHE method, as summarized in Table 4.

TABLE 4 Comparison of the proposed CLALHE versus the conventional CLAHE approach
Table 4- Comparison of the proposed CLALHE versus the conventional CLAHE approach

The novelty and advantages of CLALHE are as follows:

Multiple Enhancement Concept: We introduce a multi-enhancement process that enhances the input image multiple times. This approach helps identify the optimal enhancement parameters by evaluating each enhanced image.

Adaptive Parameter Tuning: We propose a new fitness function called the Comprehensive Image Quality Index (CIQI), which integrates PSNR, entropy, and AMBE. CIQI facilitates the evaluation and identification of the optimal parameters (i.e., CBD and I_CL) for CLALHE without requiring user expertise.

Image Subdivision: We introduce the concept of dividing the input image before applying the algorithm, allowing for local processing and better preservation of local details. This contrasts with CLAHE, which applies a uniform enhancement across the entire image.

Local Enhancement for Each Subimage: The identified optimal parameters are applied to enhance each subimage independently. This process ensures the preservation and enhancement of local details while minimizing unwanted artifacts. The enhanced subimages are then integrated to form the final image.

Real-Time Applicability: CLALHE is particularly useful for various image-processing applications, especially in real-time scenarios, due to its low computational time.

SECTION VII.Data Sample and Measurement Metrics
The proposed method was assessed using 821 sample images drawn from three distinct datasets. The first dataset, Faces 1999, comprises 450 samples with dimensions of 896×592 [85]. The second dataset, Pasadena-Houses 2000 [86], consists of 241 samples sized at 1760×1168 , and the third dataset, DIARETDB1 [87], contains 130 samples measuring 1500×1100 . These datasets were obtained from the Image Processing Place, California Institute of Technology database, and Standard Diabetic Retinopathy Database. The datasets were selected due to their diverse array of image types, which encompass various levels of information richness. Some images featured extensive uniform backgrounds, such as those depicting the sky or sea, and thus contain small yet pertinent details. Meanwhile, the others showcased intricate sections alongside uniform areas. This diversity ensured that the evaluation encompassed various levels of image quality and information richness.

The proposed CLALHE was compared with seven recently developed methods: CLAHE [28], POSHE [29], BHM [58], IAECHE [32], AEIHE [33], Auto-CLAHE [83] and ACL-CLAHE [81]. These methods were selected based on their similarities in approach to the proposed CLALHE: (i) All those methods were derived and modified from CHE. (ii) These methods employ a similar concept of dividing an image into subimages after the local enhancement each subimage contrast. (iii) These methods can preserve or enhance image details. The most recent HE-based CE methods include BHM, IAECHE, AEIHE, Auto-CLAHE and ACL-CLAHE. which were published in 2018, 2020, 2021, 2023, and 2024 respectively. The study encompassed qualitative and quantitative evaluations to appraise the efficacy of the proposed technique. The evaluation criteria included contrast, information retention, brightness, image structure and naturalism. Qualitative estimation involves scrutinising the visual quality of the final images, with a primary focus on CE, naturalness and over-enhancement. The original image’s naturalness and details must be upheld throughout the improvement process, and a favourable overall contrast must be ensured without introducing undesirable artifacts. The image’s noise level must be maintained below a specified threshold. Visual assessment proves to be a valuable criterion for the evaluation of the proposed approach. In addition, quantitative assessments using benchmarks, such as PSNR, entropy, AMBE, SSI, CII and root mean square error (RMSE), were adopted to compare the proposed approach with other CE (CHE)-based methods.

Discrete entropy (DE) serves as a metric for image quality; it estimates the amount of information present in an image [42], [88]. A high entropy image contains abundant valuable information; therefore, a positive correlation exists between image richness and entropy. Contrarily, a high noise level may occasionally influence and result in a high DE. Hence, qualitative analysis is necessary for these cases to assist in addressing high DE value identification due to noise or information richness. DE is expressed as follows:
DE=−∑l=0L−1p(l).log2(p(l))(7)
View SourceRight-click on figure for MathML and additional features.where p(l) represents the histograms’ PDF, and l denotes the grey levels of images. If the probability distribution values of all images have the same intensities, then the image entropy will be optimal, that is, p(0) = p(1) = p(2) …= p(L-1) =1/(L) [88].

The AMBE quality factor is used to measure the capacity of a technique to preserve the image’s mean brightness [19], [62]. The AMBE valve is determined through measurement of the differences between the mean brightness of the original and output images using the following equations:
AMBE=I(w,h)mean−O(w,h)mean(8)
View SourceRight-click on figure for MathML and additional features.where w and h refer to the count of pixels in the images’ rows and columns, respectively; I(w,h)mean indicates the mean of input images, and O(w,h)mean represents the mean of output images. A small difference between the original and resultant images implies that the final image preserves the brightness of the original [42].

PSNR is another key metric for the assessment of the improvement between the original and resultant images; this metric calculates the MSE to determine the degree of degradation [42], [89]. PSNR can be computed using Eq. (10):
MSEPSNR=1w×h∑w=1W∑h=1H[Ii(w,h)−Io(w,h)]2=10log10((Max(Ii))2MSE)(9)(10)
View SourceRight-click on figure for MathML and additional features.where (Max(Ii))2 , W, H, Ii(w,h) and Io(w,h) refer to the maximum grey level intensity to the original image Ii, the columns and rows of the original image, intensity of the original image pixels and intensity of the resultant image pixels, respectively.

MSE was used to quantify the mean intensity difference between the input and output images. The enhanced image exhibited less deterioration than the input image if the MSE value is minimal. Eqs. (9) and (10) show the inverse relationship of MSE and PSNR. As a result, a low MSE value decreases noise amplification, improves image quality and increases PSNR [40]. The CII assessment factor serves as a method for the evaluation of the efficacy of techniques through comparison of the contrast levels of the processed and input images [90], [91]. This comparison is expressed through the following equations:
CII=CImprovedCOriginal(11)
View SourceRight-click on figure for MathML and additional features.where CImproved and COriginal denote the average of contrast for a ROI in the processed and original images, respectively. Eq. (12) is used to compute the images’ contrast:
C=f−bf+b(12)
View SourceRight-click on figure for MathML and additional features.where f represents the mean grey level (MGL) of a specific aspect of an image, and b denotes the MGL of the surrounding area.

The SSI is an evaluation metric used to assess the likeness between two images. Its value ranges between [0, 1], which signifies a complete dissimilarity when approaching 0 and an exact match at 1. A high SSI value typically denotes a superior image quality [46], [92]. Eq. (17) is utilised to calculate SSI.
μxμyσ2xσ2ySSI(w,h)=1T∑i=1TXi=1T∑i=1TYi=1T−1∑i=1T(Xi−X¯¯¯¯)2=1T−1∑i=1T(Yi−Y¯¯¯¯)2=(2μxμy+c1)(2σxy+c2)(μ2x+μ2y+c1)(σ2x+σ2x+c2)(13)(14)(15)(16)(17)
View SourceRight-click on figure for MathML and additional features.where μx and μy represent the averages of images x and y, respectively, and σ2x , and σ2y are variances. c1 and c2 represent small constants.

RMSE serves as a metric for the disparity assessment between the original and enhanced images. In this context, the RMSE exhibits an inverse relationship with image quality. A low RMSE indicates an output image with great detail and reduced distortion compared with the original image [75]. The RMSE is calculated as follows:
RMSE=∑i=Wi=1∑j=Hj=1(I(w,h)−O(w,h))2W×H−−−−−−−−−−−−−−−−−−−−−−−−−−√(18)
View SourceRight-click on figure for MathML and additional features.where W and H represent the image rows and columns, respectively. Meanwhile,I(i,j) , and O(i,j) indicate the rows and columns of the original and resulting images, respectively.

The proposed CLALHE and all the testing methods were simulated on a personal computer with the following specifications: Intel(R) Core (TM) i7-10750H CPU @ 2.60 GHz 2.59 GHz, 16 GB RAM, SSD hard disk and Spyder (Python 3.8) as the integrated development environment.

SECTION VIII.Results and Discussion
To validate the proposed CLALHE capabilities, we conducted qualitative and quantitative assessments. The selected state-of-the-art techniques for performance evaluation included CLAHE, POSHE, BHM, IAECHE and AEIHE. The optimum parameters recommended by the respective authors of these methods were utilised in the implementation process. Qualitative analysis provided valuable insights and a comprehensive familiarity with the performance of the proposed CLALHE. We selected three datasets (i.e. faces 1999, Pasadena-Houses 2000 and DIARETDB1) to perform this analysis and obtained one image from each dataset (Figures 12, 19 and 23, respectively) to discuss the qualitative performance in detail. The selected samples were labelled as human-face, House 01 and DIARETDB1-001. Complementing the qualitative analysis, the quantitative analysis outcomes for the three images were tabulated in Tables 5–​7. The values highlighted in bold and underlined in the tables denote the best and second-best findings, respectively. Table 8 contains the average of quantitative analysis findings on all images for all examined datasets, and Table 9 includes detailed findings from the analysis of average computational times.
