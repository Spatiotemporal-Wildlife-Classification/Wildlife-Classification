# Image Classification

Please keep in mind the models are trained to form parent node classifiers within the cascading taxonomic structure. 
Due to the massive memory and computational constraints of CNN training and the availability of a single Nvidia GeForce RTX 3060 GPU unit, 
the models are required to be trained one at a time to avoid memory overloads on the CPU. 

The image classification modelling process is accomplished through the below scripts:

## [Modelling](../../src/models/image/modelling.md)
This file performs the CNN Classification model training. It creates the EfficientNet-B6 model with ImageNet pre-trained weights and 
modifies it to train the top weights on the new dataset as a form of transfer learning.
Additionally, the model training and testing over the number of training epochs is visualized and saved if required.

## [Model Validation](../../src/models/image/validation.md)
This file performs the validation of the trained CNN model on an external validation set to determine the true performance metrics of the model. 
The [classificatio report](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.classification_report.html) of the model and the 
model's balanced accuracy are written to csv files for analysis and visualization. 
Please view: `notebooks/image_classification/image_classification_visualization.ipynb`. 
The data is saved in: `notebooks/image_classification/taxon_image_classification_cache/` directory

## [Manual Classification Test](../../src/models/image/manual_check.md)
This file provides the capability to load a CNN model, and select an image to perform a classification prediction. 
This is a simple manual check to confirm the model, data, and predictions are working as expected. 
This file is not to be used extensively, but to rather provide a quick sanity check. 
The images are sourced from the `taxon_test` directory, so the model has not been trained over them before.