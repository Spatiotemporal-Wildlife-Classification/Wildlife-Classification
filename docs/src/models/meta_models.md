# Metadata Modelling

The metadata modelling process is accomplished through the below scripts:
Please keep in mind the models are trained to form parent node classifiers within the cascading taxonomic structure.

### [Pipeline](../../src/models/meta/pipeline.md)
This file performs data cleaning, transformation, and structuring for use within the metadata models.

### Model Training

### [Neural Network](../../src/models/meta/neural_network.md)
The neural network metadata model training and evaluation process. 
Hyperparameter tuning involved determining the optimal learning rate for the model due to the varying levels of abstraction
generated at different taxonomic levels. 

### Random Forest

### [XGBoost](../../src/models/meta/xgboost.md)
The XGBoost metadata model training and evaluation process. 
Hyperparameter tuning involved determining the optimal tree depth within the XGBoost model.

### [Decision Tree](../../src/models/meta/decision_tree.md)
The decision tree metadata model training and evaluation process.
Hyperparameter tuning involved determining the optimal decision tree depth for the model.

### AdaBoost

### [K-means Silhouette Score Automation](../../src/models/meta/sil_score.md)
The Silhouette score is a method of automating the selection of the number of centroids for a K-means clustering algorithms. 
This was used to determine the optimal number of centroids used to capture the geographic location distribution of at each parent node, to 
create a useful location encoding within the data. 