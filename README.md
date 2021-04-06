## Machine Learning Ensemble Methods

The projects investigates different ensemble machine learning methods for classification of images of digits using the MNIST dataset.

### Schematic of Ensemble Classifiers

Soft Voting Classifier           |  Stacking Classifier
:-------------------------:|:-------------------------:
![](images/soft_voting.png)  |  ![](images/stacking.png)

### Hyperparameter Optimization for Individual Classifier

Random Forest classifier
Soft Voting Classifier           |  Stacking Classifier
:-------------------------:|:-------------------------:
![](images/rf_Score_Estimators.png)  |  ![](images/rf_Score_MaxDepth.png)

Extra Trees Classifier
Soft Voting Classifier           |  Stacking Classifier
:-------------------------:|:-------------------------:
![](images/et_Score_Estimators.png)  |  ![](images/et_Score_MaxDepth.png)


### Accuracy Results 

	
|    Classifier Method    | Test Set Accuracy (%) |    |     Ensemble Method    | Test Set Accuracy (%) |
|:-----------------------:|:---------------------:|    |:----------------------:|:---------------------:|
|      Random Forests     |         96.87         |    | Soft Voting Classifier |         97.42         |
|      Extra Forests      |         97.07         |    |   Blending Classifier  |         97.08         |
| Support Vector Machines |         96.46         |


|     Ensemble Method    | Test Set Accuracy (%) |
|:----------------------:|:---------------------:|
| Soft Voting Classifier |         97.42         |
|   Blending Classifier  |         97.08         |
