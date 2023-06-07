# Cascading Ensemble Classifier

## Introduction
The cascading ensemble classifier capitalizes upon the determined trends within image and metadata classification 
within wildlife's taxonomic structure. 

Specifically: 

- Metadata classification performance increases with decreasing taxonomic level.

- Image classification performance decreases with decreasing taxonomic level. 

The below diagram visualizes the high level concept. 
The concept makes use of a cascading selective classifier (classifier per parent node) for both the image and
metadata classification components, resulting in dual, symmetrical cascading classification trees. 
At each level of the cascade (parent node) the trees communicate, to make a joint decision leveraging or mitigating 
their strengths or weaknesses to form a robust decision. 

<img height="382" src="../images/cec_architecture.png" width="595" alt="cascading ensemble architecture" style="display: block; margin: 0 auto"/>

## Models
The models specified below are used at each parent node within the cascading classification. 
Each model is trained on data limited to its taxonomic child nodes.

### Image Classification
The image classifier used is the [EfficientNet-B6](https://arxiv.org/pdf/1905.11946.pdf) Convolutional Neural Network (CNN). 
This CNN was selected due to its comparatively smaller size due to the efficient nature of its training and structure, 
this is essential due to the quantity of models required to be trained within the cascading ensemble classifier. 
It is acknowledged that potentially improved results could be generated from using state of the art image classifiers in the form of 
image transformers such as [Vit-G/14](https://openaccess.thecvf.com/content/CVPR2022/html/Zhai_Scaling_Vision_Transformers_CVPR_2022_paper.html)

The model uses transfer learning in the form of model weights pre-trained on the [ImageNet](https://www.image-net.org/) dataset.
The model's output is augmented to adapt to the dataset. The softmax output layer is replaced by a two-dimensional average
global pooling layer to flatten and average the output from the prior convolutional layer, followed by a softmax layer specified by the 
number of child nodes.

#### Image Classifier Specifications
- Batch size: $[4-32]$ due to variable training sizes
- Input size: $(528, 528, 3)$ with pixel values in range $[0, 255]$
- Drop connect rate of $0.2$
- Learning rate of $0.001$ with Adam optimizer
- Loss function: categorical cross-entropy

### Metadata Classification


## Joint Decision

## Limitations