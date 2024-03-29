## Machine Learning Ensemble Methods

The projects investigates different ensemble machine learning methods for classification of images of digits using the MNIST dataset.

### Schematic of Ensemble Classifiers

Soft Voting Classifier           |  Stacking Classifier
:-------------------------:|:-------------------------:
![](src_images/soft_voting.png)  |  ![](src_images/stacking.png)

### Hyperparameter Optimization for Individual Classifiers

Random Forest classifier
Number of Estimators           |  Maximum Depth of tree
:-------------------------:|:-------------------------:
<img src="src_images/rf_Score_Estimators.png" width="400" height="400"> | <img src="src_images/rf_Score_MaxDepth.png" width="400" height="400">


Extra Trees Classifier
Number of Estimators           |  Maximum Depth of tree
:-------------------------:|:-------------------------:
<img src="src_images/et_Score_Estimators.png" width="400" height="400"> | <img src="src_images/et_Score_MaxDepth.png" width="400" height="400">

### Accuracy Results 

	
|    Classifier Method    | Test Set Accuracy (%) |    
|:-----------------------:|:---------------------:|   
|      Random Forests     |         96.87         |     
|      Extra Forests      |         97.07         |    
| Support Vector Machines |         96.46         |


|     Ensemble Method    | Test Set Accuracy (%) |
|:----------------------:|:---------------------:|
| Soft Voting Classifier |         97.42         |
|   Blending Classifier  |         97.08         |
