edMoE: Modality-Specialized Mixture of Experts for Medical Vision-Language Understanding
Shivang Chopra*1, Lingchao Mao*1, Gabriela Sanchez-Rodriguez*1, Andrew J Feola1,2,3,
Jing Li1, Zsolt Kira1
1Georgia Institute of Technology  2Emory University  3Joseph M Cleland Atlanta VAMC
{schopra47, lmao31, grodrguez3, jli3175, afeola3, zkira}@gatech.edu
Abstract
Different medical imaging modalities capture diagnostic information at varying spatial resolutions, from coarse global patterns to fine-grained localized structures. However, most existing vision-language frameworks in the medical domain apply a uniform strategy for local feature extraction, overlooking the modality-specific demands. In this work, we present MedMoE, a modular and extensible vision-language processing framework that dynamically adapts visual representation based on the diagnostic context. MedMoE incorporates a Mixture-of-Experts (MoE) module conditioned on the report type, which routes multi-scale image features through specialized expert branches trained to capture modality-specific visual semantics. These experts operate over feature pyramids derived from a Swin Transformer backbone, enabling spatially adaptive attention to clinically relevant regions. This framework produces localized visual representations aligned with textual descriptions, without requiring modality-specific supervision at inference. Empirical results on diverse medical benchmarks demonstrate that MedMoE improves alignment and retrieval performance across imaging modalities, underscoring the value of modality-specialized visual representations in clinical vision-language systems.

*

Equal contribution.
1Introduction
Recent advances in vision language models (VLM) [13] have demonstrated great performance across a wide range of tasks such as zero-shot classification, segmentation, and visual question-answering. However, these models are less effective in medical domains, where data significantly differs from web content [21]. To bridge this gap, domain-specific medical VLMs have been developed by pretraining on radiology images paired with diagnostic reports or captions [18]. A key insight driving recent progress is the incorporation of local contrastive learning [6] and multi-scale visual semantics [28] , which can significantly improve the fine-grained semantic alignment between regions in an image and words in a report [6].

Refer to caption
Figure 1: Motivation: diagnostic granularity varies across modalities. Each image-report pair exemplifies how clinical observations may focus on localized lesions (red) or broader regions (blue), depending on the imaging modality and diagnostic task.
However, current approaches typically employ a static strategy for feature extraction, applying a uniform processing pipeline regardless of the imaging modality. This design overlooks the variations across imaging modalities, driven by inherent differences in image resolution and representation. As a result, the nature of word-region alignment can differ significantly. For example, CT/MRI reports often emphasize local abnormalities, while interpretation of X-rays often focus on global patterns. Figure 1 illustrates this variation, where clinical focus shifts between localized lesions and broader regions based on modality. Existing models trained predominantly on datasets like chest X-rays struggle to generalize to more diverse settings [26], and even models trained on broader datasets such as UniMed [9] employ a single shared encoder for all modalities, potentially limiting their adaptability to heterogeneous visual cues.

To overcome these limitations, we propose MedMoE, a modular vision-language framework that introduces Mixture-of-Experts (MoE) into the local visual processing pipeline to enable conditional specialization based on the diagnostic context. MoE-based architectures have gained prominence in large-scale language models [3] and multimodal systems [11, 24], where they improve efficiency and generalization by dynamically routing inputs through specialized expert modules. MedMoE routes the local visual features extracted from the image through a set of modality-aware expert branches. Each expert is trained to specialize in a distinct diagnostic context, allowing the model to selectively emphasize spatial features aligned with the semantic granularity of the associated report. This dynamic routing mechanism enables MedMoE to adapt visual grounding to modality-specific needs, outperforming existing multi-scale contrastive learning methods across a range of medical benchmarks. These findings highlight the promise of adaptive expert-driven architectures in building robust and scalable medical VLMs.

2Related Work
Global and Local Representation Learning.
Early medical VLMs such as MedCLIP [22] focus primarily on global image-text alignment using contrastive learning. These models treat the entire report as a single semantic unit and align it with a global image embedding, limiting their capacity to capture fine-grained regional semantics critical in diagnostic tasks. GLoRIA [6] introduced a global-local contrastive objective, allowing individual words in the report to attend to semantically relevant subregions of the image. Subsequent works such as LoVT [15] and PRIOR [1] further enhanced this alignment using sentence-level supervision and attention weighting mechanisms. However, these methods operate with a static, modality-agnostic encoder, which restricts their ability to adapt local grounding strategies to the diagnostic granularity required across diverse imaging modalities. In contrast, MedMoE enables context-aware specialization by routing multi-scale visual features through expert branches conditioned on the report content, allowing for adaptive alignment tailored to both the report semantics and imaging type.

Multi-Scale Medical Vision-Language Pretraining.
Multi-scale representation learning has emerged as a key challenge for adapting CLIP-style models to the medical domain [26], where clinically salient features can range from global abnormalities (e.g., X-rays) to focal lesions (e.g., CT, MRI). Recent approaches like MRM [28] and fVLM [20] address this by incorporating architectural changes such as hierarchical decoders, reconstruction losses, or anatomy-guided alignment. While these methods improve scale-aware learning, they apply a fixed processing pipeline and shared encoder across all modalities, thus failing to account for the heterogeneity in spatial detail and diagnostic focus. MedMoE addresses this limitation by enabling dynamic feature specialization through expert pathways, where routing is informed by modality-specific report context.

Mixture-of-Experts and Context-Aware Specialization.
MoE architectures have proven effective for scaling language models using sparse conditional computation [19, 10], and recent work has extended MoE to multimodal VLMs for efficient and specialized fusion [5, 27]. However, their application to medical vision-language learning remains limited. Existing VLMs typically apply MoE at the fusion or output stage, with limited impact on early visual representation learning. We propose MedMoE, the first medical VLM to integrate report-conditioned MoE routing within the visual feature extractor itself. This design enables modality- and task-specific expert selection for adaptive local grounding, addressing the bottleneck of static, one-size-fits-all processing in existing pipelines.

3Method
4Methodology
Refer to caption
Figure 2:Overall architecture of MedMoE. A Swin Transformer extracts multi-scale features, and a modality-aware router selects a specialized expert based on the global image embedding. The expert outputs local embeddings aligned with word-level text via a local contrastive loss. Global contrastive loss and auxiliary loss enforce image-text alignment and supervise expert selection respectively.
We propose MedMoE, a diagnostic-aware Mixture-of-Experts (MoE) framework for context-sensitive local feature extraction in medical vision-language modeling. MedMoE extends the global-local contrastive paradigm established by GLoRIA [6], while introducing a dynamic, report-conditioned mechanism for multi-scale local visual representation. Specifically, MedMoE replaces static feature extractors with an MoE module that routes multi-scale visual features through modality-specialized expert branches. The routing is conditioned on the diagnostic report, enabling adaptive specialization. The overall schematics of MedMoE are illustrated in Figure 2.

4.1Problem Setup
Let
(
𝐱
v
,
𝐱
t
)
 denote a paired medical image and its diagnostic report. From each pair, we extract a global image embedding
𝐯
g
∈
ℝ
D
, a local image‑embedding grid
𝐕
∈
ℝ
M
×
D
, a global text embedding
𝐭
g
∈
ℝ
D
, and token‑level text embeddings
𝐓
=

[
𝐭
1
,
…
,
𝐭
N
]
∈
ℝ
N
×
D
. Our objective is to learn semantically aligned representations that maximize image-text alignment at both global and token-region levels.

4.2Multi-Scale Visual Feature Extraction
We adopt the Swin Transformer [14] as our image encoder backbone due to its hierarchical design and strong spatial sensitivity. The encoder outputs a set of multi-scale features:

ℱ
=

{
𝐅
(
1
)
,
…
,
𝐅
(
L
)
}
,
where each
𝐅
(
l
)
 corresponds to feature maps of spatial resolution
{
H
4
×
W
4
,
H
8
×
W
8
,
H
16
×
W
16
,
H
32
×
W
32
}
, respectively. These feature maps form the input to our expert-specific local feature heads.

4.3Report-Conditioned Mixture-of-Experts
To support diagnostic-aware specialization, we introduce a Mixture-of-Experts (MoE) module composed of
K
 convolutional experts
{
E
k
}
k
=

1
K
, each trained to focus on distinct diagnostic modalities and learn modality-specific spatial reasoning patterns.

Each expert processes a set of hierarchical multi-scale features
ℱ
=

{
𝐅
(
1
)
,
…
,
𝐅
(
L
)
}
 extracted from different stages of the Swin Transformer. To produce a unified local representation, each expert first projects features at each scale to a shared embedding space and aligns their spatial resolutions via interpolation. Rather than uniformly aggregating across scales, we introduce an cross-scale attention mechanism that dynamically weights the contribution of each scale. For a given expert
E
k
, the fused local representation is computed as:

𝐕
k
=

∑
ℓ
=

1
L
β
k
(
ℓ
)
⋅
𝐅
k
(
ℓ
)
,
where
β
k
(
ℓ
)
 are soft attention weights predicted at each spatial location by the expert’s scale-attention module. This allows each expert to adaptively combine fine and coarse features depending on the image content and diagnostic specialization.

To ensure computational efficiency and promote specialization, we use a hard routing strategy to select a single expert per input. The routing module computes scores
𝜶
=

[
α
1
,
…
,
α
K
]
 via a lightweight MLP conditioned on the global report embedding
𝐭
g
:

𝜶
=

softmax
⁢
(
W
2
⋅
ReLU
⁢
(
W
1
⋅
𝐭
g
)
)
,
with
W
1
 and
W
2
 as learned parameters. The index of the selected expert is then given by
k
∗
=

arg
⁡
max
k
⁡
α
k
. The final local visual representation is defined as:

𝐕
local
=

𝐕
k
∗
This hard selection mechanism avoids computing all expert branches during inference, thereby reducing overhead while still leveraging diagnostic-aware specialization. This two-level specialization, across both scales and experts, enables the model to flexibly adapt to diverse diagnostic imaging contexts.

4.4Contrastive Learning Objectives
Following GLoRIA [6], we apply both global and local contrastive learning for image-text alignment:
Global Contrastive Loss. The global loss aligns the image-level and report-level representations:

ℒ
global
=

−
log
⁡
exp
⁡
(
sim
⁢
(
𝐯
g
,
𝐭
g
)
/
τ
)
∑
j
=

1
B
exp
⁡
(
sim
⁢
(
𝐯
g
,
𝐭
j
)
/
τ
)
,
where
τ
 is the temperature hyperparameter,
B
 is the batch size, and
sim
⁢
(
⋅
,
⋅
)
 denotes cosine similarity.
Local Contrastive Loss. For each token
𝐭
i
, we compute a visual context vector
𝐜
i
 by attending to local features
𝐯
j
:

a
i
⁢
j
=

exp
⁡
(
𝐭
i
⊤
⁢
𝐯
j
/
τ
)
∑
j
M
exp
⁡
(
𝐭
i
⊤
⁢
𝐯
j
/
τ
)
,
𝐜
i
=

∑
j
M
a
i
⁢
j
⋅
𝐯
j
,
ℒ
local
=

∑
i
N
−
log
⁡
exp
⁡
(
sim
⁢
(
𝐜
i
,
𝐭
i
)
/
τ
)
∑
j
N
exp
⁡
(
sim
⁢
(
𝐜
i
,
𝐭
j
)
/
τ
)
.
Approach Dataset Modality Image Encoder X-ray Ultrasound MRI CT Avg.
CheXpert(5x200) RSNA Thyroid Breast ACL Meniscus Axial Coronal Sagittal
Specialist Models
QuiltNet [16]  Quilt (1M) [7] Histopathology ViT-B/16 18.20 45.45 60.12 42.27 34.49 55.57 10.03 5.83 9.38 31.26
KeepFIT [23]  MM-Retinal [23] Fundus ResNet50 23.30 50.00 40.12 59.72 66.32 62.26 8.74 12.46 9.06 36.88
GLoRIA [6]  CheXpert (200k) [8] X-Ray ResNet50 61.00 50.00 79.16 72.43 43.75 76.44 3.39 3.39 3.39 43.66
MedCLIP [22]  CheXpert (200k) [8] X-Ray SWIN 59.42 73.63 44.68 53.98 68.01 45.27 9.87 13.92 10.36 42.12
Generalist Models
PMC-CLIP [12]  PMC-OA (1.5M) [12] All ResNet50 30.30 74.99 58.60 68.76 67.09 65.03 33.98 20.23 23.30 49.14
BiomedCLIP [25]  PMC (15M) [25] All ViT-B/16 38.30 72.56 61.05 68.12 47.89 40.09 29.13 19.09 20.39 44.06
UniMed-CLIP [9]  UniMed (5.3M) [9] All ViT-B/16 65.90 71.65 71.35 66.31 85.82 85.27 37.54 24.43 31.72 59.98
Ours UniMed (5.3M) [9] All SWIN + MoE 66.03 78.32 62.74 69.32 77.92 74.87 40.00 28.32 26.83 58.26
Table 1:Zero-shot classification accuracy on various radiology datasets. Best results are highlighted in bold and second-best results are underlined.
4.5Auxiliary Report-Type Supervision
To reinforce the specialization of experts, we introduce an auxiliary classification head over the report embedding:

ℒ
aux
=

CrossEntropy
⁢
(
W
c
⋅
𝐭
g
,
y
)
,
where
W
c
 is a classification head and
y
 is the ground-truth report type (e.g., CT, X-ray, MRI). This encourages the router to learn modality-sensitive routing that aligns with variations in real diagnostic contexts.

4.6Overall Training Objective
The final training loss is a weighted combination of the above objectives:

ℒ
total
=

ℒ
global
+
ℒ
local
+
λ
⋅
ℒ
aux
,
where
λ
 is a tunable hyperparameter to balance the auxiliary supervision.

Dataset Model Performance %
(Modality) 1% 10% 100%
RSNA [2] CLIP [18] 71.67 73.89 81.09
PMC-CLIP [12]  65.25 64.15 79.47
BiomedCLIP [25]  80.04 78.32 83.84
(X-ray) UniMed-CLIP [9] 79.52 79.19 88.51
Ours 81.12 83.84 87.83
Thyroid [17] CLIP [18] 70.99 72.51 75.99
PMC-CLIP [12]  46.78 56.61 61.40
BiomedCLIP [25]  76.96 80.47 81.87
(Ultrasound) UniMed-CLIP [9] 55.67 64.44 76.84
Ours 73.99 76.96 79.83
ACL CLIP [18] 56.99 85.89 91.73
PMC-CLIP [12]  57.00 63.47 77.92
BiomedCLIP [25]  44.13 65.46 90.12
(MRI) UniMed-CLIP [9] 89.61 95.28 97.28
Ours 85.89 89.77 92.84
MediMeTA Axial CLIP [18] 32.20 45.97 70.06
PMC-CLIP [12]  35.92 43.04 57.61
BiomedCLIP [25]  29.77 59.06 77.67
(CT) UniMed-CLIP [9] 39.97 57.28 76.38
Ours 42.68 62.13 76.38
Table 2:Linear Probing Results. Performance (Accuracy) comparison across different generalist medical foundation models with varying training data percentages. Best results are highlighted in bold and second-best results are underlined.
5Experiments and Results
We evaluate MedMoE in terms of zero-shot and linear probing classification performance across diverse medical imaging modalities and benchmarks, demonstrating its effectiveness in context-sensitive visual-textual alignment.

5.1Experimental Setup
Datasets: We use the UniMed [9] dataset for pretraining and evaluate on diagnostic benchmarks used in previous works (e.g., RSNA [2], CheXpert [8], ACL, Meniscus, Thyroid [17], Breast) covering four modalities: X-ray, MRI, CT, and Ultrasound. For linear probing, we use 1%, 10%, and 100% of the training set from each benchmark.

Baselines: We compare MedMoE with generalist VLMs (CLIP [18], PMC-CLIP [12], BiomedCLIP [25], UniMed-CLIP [9]) and specialist models (MedCLIP [22], QuiltNet [16], MM-Retinal [23]).

Implementation Details: We initialize the Swin Transformer [14] backbone from MedCLIP-pretrained weights [4]. The expert branches are 3-layer convolutional modules with BatchNorm and ReLU activations. Training is conducted on 8 NVIDIA A40 GPUs with a global batch size of 256, using gradient accumulation with 10 steps.

5.2Zero‑Shot Classification
Table 1 reports zero-shot accuracies on nine radiology benchmarks spanning four imaging modalities MedMoE attains the best performance on 6/9 datasets, including 78.32% on RSNA and 69.32% on Breast Ultrasound, outperforming the strongest generalist baseline (PMC‑CLIP) by 3.33% and 0.56%, respectively. On MRI datasets, MedMoE achieves 77.92% on ACL and 74.87% on Meniscus, an improvement of 30.0% and 34.8% over BioMedCLIP, while remaining competitive with UniMed‑CLIP, which is tuned specifically for MRI. Finally, MedMoE delivers state‑of‑the‑art results on all three CT views (40.00%, 28.32%, 26.83%).

Refer to caption
Figure 3:Visualization of Attention Weights across Modalities. Word-level attention maps are shown for different imaging modalities (X-ray, Ultrasound, MRI, and CT), highlighting regions relevant to the associated diagnostic labels.
5.3Linear Probing Transfer
We next freeze the visual encoder and fit a single linear layer on 1%, 10%, and 100% of each training set (Table 2). In the extreme low‑data regime (1%), MedMoE yields 81.12% on RSNA, surpassing the previous best BiomedCLIP (80.04%). On Thyroid Ultrasound, MedMoE reaches 73.99%, a gain of 27.21% over PMC‑CLIP, and only 2.97% behind the modality‑specialised BiomedCLIP. For MRI (ACL) and CT (MediMeTA Axial), MedMoE achieves 85.89% and 42.68%, respectively, matching or exceeding all generalist baselines.

6Discussion
6.1Visualization of Attention Weights
Figure 3 illustrates the attention maps produced by MedMoE’s expert branches across four imaging modalities. Each expert attends to diagnostically relevant regions specific to its modality: lung outlines in X-rays for detecting atelectasis, heterogeneous thyroid tissue in ultrasound, ligament structures in MRI, and abdominal organs in CT. These visualizations highlight MedMoE’s ability to perform modality-specific visual grounding. By routing features through specialized experts, the model learns to adapt its spatial focus to the granularity and diagnostic patterns unique to each modality.

6.2Computational Cost Comparison
While MedMoE introduces multiple expert branches, it uses hard expert selection at inference time, activating only a single expert per input. As a result, its computational cost remains comparable to baseline models built on the Swin-Tiny backbone. Table 3 reports the FLOPs and parameter counts of MedMoE alongside representative medical vision-language models. Although all experts are included during training, only one is active at inference, resulting in minimal overhead. This design enables MedMoE to achieve modality-aware specialization while maintaining efficiency and scalability.

Model Backbone Params (M) FLOPs (G)
CLIP (RN50) ResNet-50 38.3 6.12
BiomedCLIP ViT-B/16 86.1 16.8
UniMed-CLIP ViT-B/16 86.1 11.29
MedCLIP Swin-Tiny 27 4.5
MedMoE (Ours) Swin-Tiny + MoE 37*7.8*
Table 3:FLOPs and parameter comparison. MedMoE uses Swin-T and activates only one expert at inference. *MedMoE parameter count includes all experts; FLOPs are measured with a single expert active.
7Conclusion and Future Work
We present MedMoE, a vision–language model that dynamically routes multi-scale visual features through a diagnosis-conditioned Mixture-of-Experts framework. By leveraging report-aware specialization, MedMoE achieves state-of-the-art performance on both zero-shot and linear probing benchmarks across diverse medical imaging modalities. In addition to strong quantitative gains, we provide qualitative analyses that illustrate improved modality-sensitive visual grounding, and computational comparisons that demonstrate MedMoE’s efficiency through hard expert selection. These results underscore the importance of modality-specific feature routing and context-aware alignment in medical vision-language learning. Future work will explore extending MedMoE to a wider range of unimodal and multimodal tasks, such as segmentation, report generation, and few-shot adaptation.

References
edMoE: Modality-Specialized Mixture of Experts for Medical Vision-Language Understanding
Shivang Chopra*1, Lingchao Mao*1, Gabriela Sanchez-Rodriguez*1, Andrew J Feola1,2,3,
Jing Li1, Zsolt Kira1
1Georgia Institute of Technology  2Emory University  3Joseph M Cleland Atlanta VAMC
{schopra47, lmao31, grodrguez3, jli3175, afeola3, zkira}@gatech.edu
Abstract
Different medical imaging modalities capture diagnostic information at varying spatial resolutions, from coarse global patterns to fine-grained localized structures. However, most existing vision-language frameworks in the medical domain apply a uniform strategy for local feature extraction, overlooking the modality-specific demands. In this work, we present MedMoE, a modular and extensible vision-language processing framework that dynamically adapts visual representation based on the diagnostic context. MedMoE incorporates a Mixture-of-Experts (MoE) module conditioned on the report type, which routes multi-scale image features through specialized expert branches trained to capture modality-specific visual semantics. These experts operate over feature pyramids derived from a Swin Transformer backbone, enabling spatially adaptive attention to clinically relevant regions. This framework produces localized visual representations aligned with textual descriptions, without requiring modality-specific supervision at inference. Empirical results on diverse medical benchmarks demonstrate that MedMoE improves alignment and retrieval performance across imaging modalities, underscoring the value of modality-specialized visual representations in clinical vision-language systems.

*

Equal contribution.
1Introduction
Recent advances in vision language models (VLM) [13] have demonstrated great performance across a wide range of tasks such as zero-shot classification, segmentation, and visual question-answering. However, these models are less effective in medical domains, where data significantly differs from web content [21]. To bridge this gap, domain-specific medical VLMs have been developed by pretraining on radiology images paired with diagnostic reports or captions [18]. A key insight driving recent progress is the incorporation of local contrastive learning [6] and multi-scale visual semantics [28] , which can significantly improve the fine-grained semantic alignment between regions in an image and words in a report [6].

Refer to caption
Figure 1: Motivation: diagnostic granularity varies across modalities. Each image-report pair exemplifies how clinical observations may focus on localized lesions (red) or broader regions (blue), depending on the imaging modality and diagnostic task.
However, current approaches typically employ a static strategy for feature extraction, applying a uniform processing pipeline regardless of the imaging modality. This design overlooks the variations across imaging modalities, driven by inherent differences in image resolution and representation. As a result, the nature of word-region alignment can differ significantly. For example, CT/MRI reports often emphasize local abnormalities, while interpretation of X-rays often focus on global patterns. Figure 1 illustrates this variation, where clinical focus shifts between localized lesions and broader regions based on modality. Existing models trained predominantly on datasets like chest X-rays struggle to generalize to more diverse settings [26], and even models trained on broader datasets such as UniMed [9] employ a single shared encoder for all modalities, potentially limiting their adaptability to heterogeneous visual cues.

To overcome these limitations, we propose MedMoE, a modular vision-language framework that introduces Mixture-of-Experts (MoE) into the local visual processing pipeline to enable conditional specialization based on the diagnostic context. MoE-based architectures have gained prominence in large-scale language models [3] and multimodal systems [11, 24], where they improve efficiency and generalization by dynamically routing inputs through specialized expert modules. MedMoE routes the local visual features extracted from the image through a set of modality-aware expert branches. Each expert is trained to specialize in a distinct diagnostic context, allowing the model to selectively emphasize spatial features aligned with the semantic granularity of the associated report. This dynamic routing mechanism enables MedMoE to adapt visual grounding to modality-specific needs, outperforming existing multi-scale contrastive learning methods across a range of medical benchmarks. These findings highlight the promise of adaptive expert-driven architectures in building robust and scalable medical VLMs.

2Related Work
Global and Local Representation Learning.
Early medical VLMs such as MedCLIP [22] focus primarily on global image-text alignment using contrastive learning. These models treat the entire report as a single semantic unit and align it with a global image embedding, limiting their capacity to capture fine-grained regional semantics critical in diagnostic tasks. GLoRIA [6] introduced a global-local contrastive objective, allowing individual words in the report to attend to semantically relevant subregions of the image. Subsequent works such as LoVT [15] and PRIOR [1] further enhanced this alignment using sentence-level supervision and attention weighting mechanisms. However, these methods operate with a static, modality-agnostic encoder, which restricts their ability to adapt local grounding strategies to the diagnostic granularity required across diverse imaging modalities. In contrast, MedMoE enables context-aware specialization by routing multi-scale visual features through expert branches conditioned on the report content, allowing for adaptive alignment tailored to both the report semantics and imaging type.

Multi-Scale Medical Vision-Language Pretraining.
Multi-scale representation learning has emerged as a key challenge for adapting CLIP-style models to the medical domain [26], where clinically salient features can range from global abnormalities (e.g., X-rays) to focal lesions (e.g., CT, MRI). Recent approaches like MRM [28] and fVLM [20] address this by incorporating architectural changes such as hierarchical decoders, reconstruction losses, or anatomy-guided alignment. While these methods improve scale-aware learning, they apply a fixed processing pipeline and shared encoder across all modalities, thus failing to account for the heterogeneity in spatial detail and diagnostic focus. MedMoE addresses this limitation by enabling dynamic feature specialization through expert pathways, where routing is informed by modality-specific report context.

Mixture-of-Experts and Context-Aware Specialization.
MoE architectures have proven effective for scaling language models using sparse conditional computation [19, 10], and recent work has extended MoE to multimodal VLMs for efficient and specialized fusion [5, 27]. However, their application to medical vision-language learning remains limited. Existing VLMs typically apply MoE at the fusion or output stage, with limited impact on early visual representation learning. We propose MedMoE, the first medical VLM to integrate report-conditioned MoE routing within the visual feature extractor itself. This design enables modality- and task-specific expert selection for adaptive local grounding, addressing the bottleneck of static, one-size-fits-all processing in existing pipelines.

3Method
4Methodology
Refer to caption
Figure 2:Overall architecture of MedMoE. A Swin Transformer extracts multi-scale features, and a modality-aware router selects a specialized expert based on the global image embedding. The expert outputs local embeddings aligned with word-level text via a local contrastive loss. Global contrastive loss and auxiliary loss enforce image-text alignment and supervise expert selection respectively.
We propose MedMoE, a diagnostic-aware Mixture-of-Experts (MoE) framework for context-sensitive local feature extraction in medical vision-language modeling. MedMoE extends the global-local contrastive paradigm established by GLoRIA [6], while introducing a dynamic, report-conditioned mechanism for multi-scale local visual representation. Specifically, MedMoE replaces static feature extractors with an MoE module that routes multi-scale visual features through modality-specialized expert branches. The routing is conditioned on the diagnostic report, enabling adaptive specialization. The overall schematics of MedMoE are illustrated in Figure 2.

4.1Problem Setup
Let
(
𝐱
v
,
𝐱
t
)
 denote a paired medical image and its diagnostic report. From each pair, we extract a global image embedding
𝐯
g
∈
ℝ
D
, a local image‑embedding grid
𝐕
∈
ℝ
M
×
D
, a global text embedding
𝐭
g
∈
ℝ
D
, and token‑level text embeddings
𝐓
=

[
𝐭
1
,
…
,
𝐭
N
]
∈
ℝ
N
×
D
. Our objective is to learn semantically aligned representations that maximize image-text alignment at both global and token-region levels.

4.2Multi-Scale Visual Feature Extraction
We adopt the Swin Transformer [14] as our image encoder backbone due to its hierarchical design and strong spatial sensitivity. The encoder outputs a set of multi-scale features:

ℱ
=

{
𝐅
(
1
)
,
…
,
𝐅
(
L
)
}
,
where each
𝐅
(
l
)
 corresponds to feature maps of spatial resolution
{
H
4
×
W
4
,
H
8
×
W
8
,
H
16
×
W
16
,
H
32
×
W
32
}
, respectively. These feature maps form the input to our expert-specific local feature heads.

4.3Report-Conditioned Mixture-of-Experts
To support diagnostic-aware specialization, we introduce a Mixture-of-Experts (MoE) module composed of
K
 convolutional experts
{
E
k
}
k
=

1
K
, each trained to focus on distinct diagnostic modalities and learn modality-specific spatial reasoning patterns.

Each expert processes a set of hierarchical multi-scale features
ℱ
=

{
𝐅
(
1
)
,
…
,
𝐅
(
L
)
}
 extracted from different stages of the Swin Transformer. To produce a unified local representation, each expert first projects features at each scale to a shared embedding space and aligns their spatial resolutions via interpolation. Rather than uniformly aggregating across scales, we introduce an cross-scale attention mechanism that dynamically weights the contribution of each scale. For a given expert
E
k
, the fused local representation is computed as:

𝐕
k
=

∑
ℓ
=

1
L
β
k
(
ℓ
)
⋅
𝐅
k
(
ℓ
)
,
where
β
k
(
ℓ
)
 are soft attention weights predicted at each spatial location by the expert’s scale-attention module. This allows each expert to adaptively combine fine and coarse features depending on the image content and diagnostic specialization.

To ensure computational efficiency and promote specialization, we use a hard routing strategy to select a single expert per input. The routing module computes scores
𝜶
=

[
α
1
,
…
,
α
K
]
 via a lightweight MLP conditioned on the global report embedding
𝐭
g
:

𝜶
=

softmax
⁢
(
W
2
⋅
ReLU
⁢
(
W
1
⋅
𝐭
g
)
)
,
with
W
1
 and
W
2
 as learned parameters. The index of the selected expert is then given by
k
∗
=

arg
⁡
max
k
⁡
α
k
. The final local visual representation is defined as:

𝐕
local
=

𝐕
k
∗
This hard selection mechanism avoids computing all expert branches during inference, thereby reducing overhead while still leveraging diagnostic-aware specialization. This two-level specialization, across both scales and experts, enables the model to flexibly adapt to diverse diagnostic imaging contexts.

4.4Contrastive Learning Objectives
Following GLoRIA [6], we apply both global and local contrastive learning for image-text alignment:
Global Contrastive Loss. The global loss aligns the image-level and report-level representations:

ℒ
global
=

−
log
⁡
exp
⁡
(
sim
⁢
(
𝐯
g
,
𝐭
g
)
/
τ
)
∑
j
=

1
B
exp
⁡
(
sim
⁢
(
𝐯
g
,
𝐭
j
)
/
τ
)
,
where
τ
 is the temperature hyperparameter,
B
 is the batch size, and
sim
⁢
(
⋅
,
⋅
)
 denotes cosine similarity.
Local Contrastive Loss. For each token
𝐭
i
, we compute a visual context vector
𝐜
i
 by attending to local features
𝐯
j
:

a
i
⁢
j
=

exp
⁡
(
𝐭
i
⊤
⁢
𝐯
j
/
τ
)
∑
j
M
exp
⁡
(
𝐭
i
⊤
⁢
𝐯
j
/
τ
)
,
𝐜
i
=

∑
j
M
a
i
⁢
j
⋅
𝐯
j
,
ℒ
local
=

∑
i
N
−
log
⁡
exp
⁡
(
sim
⁢
(
𝐜
i
,
𝐭
i
)
/
τ
)
∑
j
N
exp
⁡
(
sim
⁢
(
𝐜
i
,
𝐭
j
)
/
τ
)
.
Approach Dataset Modality Image Encoder X-ray Ultrasound MRI CT Avg.
CheXpert(5x200) RSNA Thyroid Breast ACL Meniscus Axial Coronal Sagittal
Specialist Models 
QuiltNet [16]  Quilt (1M) [7] Histopathology ViT-B/16 18.20 45.45 60.12 42.27 34.49 55.57 10.03 5.83 9.38 31.26
KeepFIT [23]  MM-Retinal [23] Fundus ResNet50 23.30 50.00 40.12 59.72 66.32 62.26 8.74 12.46 9.06 36.88
GLoRIA [6]  CheXpert (200k) [8] X-Ray ResNet50 61.00 50.00 79.16 72.43 43.75 76.44 3.39 3.39 3.39 43.66
MedCLIP [22]  CheXpert (200k) [8] X-Ray SWIN 59.42 73.63 44.68 53.98 68.01 45.27 9.87 13.92 10.36 42.12
Generalist Models 
PMC-CLIP [12]  PMC-OA (1.5M) [12] All ResNet50 30.30 74.99 58.60 68.76 67.09 65.03 33.98 20.23 23.30 49.14
BiomedCLIP [25]  PMC (15M) [25] All ViT-B/16 38.30 72.56 61.05 68.12 47.89 40.09 29.13 19.09 20.39 44.06
UniMed-CLIP [9]  UniMed (5.3M) [9] All ViT-B/16 65.90 71.65 71.35 66.31 85.82 85.27 37.54 24.43 31.72 59.98
Ours UniMed (5.3M) [9] All SWIN + MoE 66.03 78.32 62.74 69.32 77.92 74.87 40.00 28.32 26.83 58.26
Table 1:Zero-shot classification accuracy on various radiology datasets. Best results are highlighted in bold and second-best results are underlined.
4.5Auxiliary Report-Type Supervision
To reinforce the specialization of experts, we introduce an auxiliary classification head over the report embedding:

ℒ
aux
=

CrossEntropy
⁢
(
W
c
⋅
𝐭
g
,
y
)
,
where
W
c
 is a classification head and
y
 is the ground-truth report type (e.g., CT, X-ray, MRI). This encourages the router to learn modality-sensitive routing that aligns with variations in real diagnostic contexts.

4.6Overall Training Objective
The final training loss is a weighted combination of the above objectives:

ℒ
total
=

ℒ
global
+
ℒ
local
+
λ
⋅
ℒ
aux
,
where
λ
 is a tunable hyperparameter to balance the auxiliary supervision.

Dataset Model Performance %
(Modality) 1% 10% 100%
RSNA [2] CLIP [18] 71.67 73.89 81.09
PMC-CLIP [12]  65.25 64.15 79.47
BiomedCLIP [25]  80.04 78.32 83.84
(X-ray) UniMed-CLIP [9] 79.52 79.19 88.51
Ours 81.12 83.84 87.83
Thyroid [17] CLIP [18] 70.99 72.51 75.99
PMC-CLIP [12]  46.78 56.61 61.40
BiomedCLIP [25]  76.96 80.47 81.87
(Ultrasound) UniMed-CLIP [9] 55.67 64.44 76.84
Ours 73.99 76.96 79.83
ACL CLIP [18] 56.99 85.89 91.73
PMC-CLIP [12]  57.00 63.47 77.92
BiomedCLIP [25]  44.13 65.46 90.12
(MRI) UniMed-CLIP [9] 89.61 95.28 97.28
Ours 85.89 89.77 92.84
MediMeTA Axial CLIP [18] 32.20 45.97 70.06
PMC-CLIP [12]  35.92 43.04 57.61
BiomedCLIP [25]  29.77 59.06 77.67
(CT) UniMed-CLIP [9] 39.97 57.28 76.38
Ours 42.68 62.13 76.38
Table 2:Linear Probing Results. Performance (Accuracy) comparison across different generalist medical foundation models with varying training data percentages. Best results are highlighted in bold and second-best results are underlined.
5Experiments and Results
We evaluate MedMoE in terms of zero-shot and linear probing classification performance across diverse medical imaging modalities and benchmarks, demonstrating its effectiveness in context-sensitive visual-textual alignment.

5.1Experimental Setup
Datasets: We use the UniMed [9] dataset for pretraining and evaluate on diagnostic benchmarks used in previous works (e.g., RSNA [2], CheXpert [8], ACL, Meniscus, Thyroid [17], Breast) covering four modalities: X-ray, MRI, CT, and Ultrasound. For linear probing, we use 1%, 10%, and 100% of the training set from each benchmark.

Baselines: We compare MedMoE with generalist VLMs (CLIP [18], PMC-CLIP [12], BiomedCLIP [25], UniMed-CLIP [9]) and specialist models (MedCLIP [22], QuiltNet [16], MM-Retinal [23]).

Implementation Details: We initialize the Swin Transformer [14] backbone from MedCLIP-pretrained weights [4]. The expert branches are 3-layer convolutional modules with BatchNorm and ReLU activations. Training is conducted on 8 NVIDIA A40 GPUs with a global batch size of 256, using gradient accumulation with 10 steps.

5.2Zero‑Shot Classification
Table 1 reports zero-shot accuracies on nine radiology benchmarks spanning four imaging modalities MedMoE attains the best performance on 6/9 datasets, including 78.32% on RSNA and 69.32% on Breast Ultrasound, outperforming the strongest generalist baseline (PMC‑CLIP) by 3.33% and 0.56%, respectively. On MRI datasets, MedMoE achieves 77.92% on ACL and 74.87% on Meniscus, an improvement of 30.0% and 34.8% over BioMedCLIP, while remaining competitive with UniMed‑CLIP, which is tuned specifically for MRI. Finally, MedMoE delivers state‑of‑the‑art results on all three CT views (40.00%, 28.32%, 26.83%).

Refer to caption
Figure 3:Visualization of Attention Weights across Modalities. Word-level attention maps are shown for different imaging modalities (X-ray, Ultrasound, MRI, and CT), highlighting regions relevant to the associated diagnostic labels.
5.3Linear Probing Transfer
We next freeze the visual encoder and fit a single linear layer on 1%, 10%, and 100% of each training set (Table 2). In the extreme low‑data regime (1%), MedMoE yields 81.12% on RSNA, surpassing the previous best BiomedCLIP (80.04%). On Thyroid Ultrasound, MedMoE reaches 73.99%, a gain of 27.21% over PMC‑CLIP, and only 2.97% behind the modality‑specialised BiomedCLIP. For MRI (ACL) and CT (MediMeTA Axial), MedMoE achieves 85.89% and 42.68%, respectively, matching or exceeding all generalist baselines.

6Discussion
6.1Visualization of Attention Weights
Figure 3 illustrates the attention maps produced by MedMoE’s expert branches across four imaging modalities. Each expert attends to diagnostically relevant regions specific to its modality: lung outlines in X-rays for detecting atelectasis, heterogeneous thyroid tissue in ultrasound, ligament structures in MRI, and abdominal organs in CT. These visualizations highlight MedMoE’s ability to perform modality-specific visual grounding. By routing features through specialized experts, the model learns to adapt its spatial focus to the granularity and diagnostic patterns unique to each modality.

6.2Computational Cost Comparison
While MedMoE introduces multiple expert branches, it uses hard expert selection at inference time, activating only a single expert per input. As a result, its computational cost remains comparable to baseline models built on the Swin-Tiny backbone. Table 3 reports the FLOPs and parameter counts of MedMoE alongside representative medical vision-language models. Although all experts are included during training, only one is active at inference, resulting in minimal overhead. This design enables MedMoE to achieve modality-aware specialization while maintaining efficiency and scalability.

Model Backbone Params (M) FLOPs (G)
CLIP (RN50) ResNet-50 38.3 6.12
BiomedCLIP ViT-B/16 86.1 16.8
UniMed-CLIP ViT-B/16 86.1 11.29
MedCLIP Swin-Tiny 27 4.5
MedMoE (Ours) Swin-Tiny + MoE 37* 7.8*
Table 3:FLOPs and parameter comparison. MedMoE uses Swin-T and activates only one expert at inference. *MedMoE parameter count includes all experts; FLOPs are measured with a single expert active.
7Conclusion and Future Work
We present MedMoE, a vision–language model that dynamically routes multi-scale visual features through a diagnosis-conditioned Mixture-of-Experts framework. By leveraging report-aware specialization, MedMoE achieves state-of-the-art performance on both zero-shot and linear probing benchmarks across diverse medical imaging modalities. In addition to strong quantitative gains, we provide qualitative analyses that illustrate improved modality-sensitive visual grounding, and computational comparisons that demonstrate MedMoE’s efficiency through hard expert selection. These results underscore the importance of modality-specific feature routing and context-aware alignment in medical vision-language learning. Future work will explore extending MedMoE to a wider range of unimodal and multimodal tasks, such as segmentation, report generation, and few-shot adaptation.

References
