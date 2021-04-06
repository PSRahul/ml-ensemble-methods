## Machine Learning Ensemble Methods

The projects investigates different ensemble machine learning methods for classification of images of digits using the MNIST dataset.

An soft-voting classifier was trained based on Random Forests and Support Vector Machine for the MNIST dataset. The optimised model achieved an test accuracy of 97.42%

A stacking classifier was also trained based on Random forests, extremely randomized Forests and Support Vector Machine that classified a test accuracy of 97.08%


Based on the exercise from “Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow", 2nd Edition, by Aurélien Géron (O’Reilly)

### Accuracy Results 
	
|    Classifier Method    | Test Set Accuracy (%) |
|:-----------------------:|:---------------------:|
|      Random Forests     |         96.87         |
|      Extra Forests      |         97.07         |
| Support Vector Machines |         96.46         |

		Ensemble Methods

|     Ensemble Method    | Test Set Accuracy (%) |
|:----------------------:|:---------------------:|
| Soft Voting Classifier |         97.42         |
|   Blending Classifier  |         97.08         |
