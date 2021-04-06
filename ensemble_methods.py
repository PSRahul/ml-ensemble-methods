from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.svm import SVC
from sklearn.ensemble import VotingClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import PredefinedSplit
from sklearn.svm import LinearSVC
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

def model_eval(model):
    print("Predicting 1")
    model_pred_val=model.predict(x_val)
    print("Predicting 2")
    model_pred_test=model.predict(x_test)
    model_val=accuracy_score(y_val,model_pred_val)
    model_test=accuracy_score(y_test,model_pred_test)
    return model_pred_test,model_pred_val,model_val,model_test

#Loading Data and splitting data into train, validation and test set
import idx2numpy
import numpy as np
file="t10k-images-idx3-ubyte"
x_test=idx2numpy.convert_from_file(file)
file="t10k-labels.idx1-ubyte"
y_test=idx2numpy.convert_from_file(file)
file="train-images-idx3-ubyte"
x_train_val=idx2numpy.convert_from_file(file)
file="train-labels-idx1-ubyte"
y_train_val=idx2numpy.convert_from_file(file)

test_fold=np.zeros((60000,1))
from sklearn.model_selection import StratifiedShuffleSplit
sss=StratifiedShuffleSplit(n_splits=1,test_size=10000)
for train_index,test_index in sss.split(x_train_val,y_train_val):
    x_train,y_train=x_train_val[train_index],y_train_val[train_index]
    x_val,y_val=x_train_val[test_index],y_train_val[test_index]
    test_fold[train_index]=-1 
    test_fold[test_index]=0
print("Training Set   ",x_train.shape,y_train.shape)
print("Validation Set ",x_val.shape,y_val.shape)
print("Test Set       ",x_test.shape,y_test.shape)

x_train=x_train.reshape(50000,784)
x_val=x_val.reshape(10000,784)
x_test=x_test.reshape(10000,784)
x_train_val=x_train_val.reshape(60000,784)    

#Renormalizing the features of the data
scal=StandardScaler()
scal.fit(x_train)
x_train=scal.transform(x_train)
x_val=scal.transform(x_val)
x_test=scal.transform(x_test)
x_train_val=scal.transform(x_train_val)

ps=PredefinedSplit(test_fold)

#Fitting the individual base classifiers on the dataset
rfc=RandomForestClassifier(n_estimators=200,max_depth=20,n_jobs=-1)
etc=ExtraTreesClassifier(n_estimators=200,max_depth=20,n_jobs=-1)
svc=SVC(kernel="rbf",probability=True,cache_size=4000)
print("Fitting RFC")
rfc.fit(x_train,y_train)
print("Fitting ETC")
etc.fit(x_train,y_train)
print("Fitting SVC")
svc.fit(x_train,y_train)

#Evaluating the base models on the validation and test data
rfc_pred_test,rfc_pred_val,rfc_val,rfc_test=model_eval(rfc)
etc_pred_test,etc_pred_val,etc_val,etc_test=model_eval(etc)
svc_pred_test,svc_pred_val,svc_val,svc_test=model_eval(svc)

#Defining the Blending Classifier
stack_train_x=np.zeros((10000,3))
stack_train_x[:,0]=rfc_pred_val
stack_train_x[:,1]=etc_pred_val
stack_train_x[:,2]=svc_pred_val
stack_train_y=y_val
rfc_blend=RandomForestClassifier(n_estimators=200,max_depth=20,n_jobs=-1)
rfc_blend.fit(stack_train_x,stack_train_y)

stack_test_x=np.zeros((10000,3))
stack_test_x[:,0]=rfc_pred_test
stack_test_x[:,1]=etc_pred_test
stack_test_x[:,2]=svc_pred_test
stack_test_y=y_test
stack_test_pred=rfc_blend.predict(stack_test_x)
blend_scr=accuracy_score(stack_test_y,stack_test_pred)
print('Blending Score',blend_scr)

print('Random Forest Classifier : Validation Set ',rfc_val,'Test Set ',rfc_test)
print('Extra Forest Classifier : Validation Set ',etc_val,'Test Set ',etc_test)
print('Support Vector Classifier : Validation Set ',svc_val,'Test Set ',svc_test)

#Defining the soft voting classifier


soft_vc=VotingClassifier(estimators=[('rf',rfc),('et',etc),('sv',svc)],voting='soft')
soft_vc_pred,soft_vc_val,soft_vc_test=model_eval(soft_vc)
print('Random Forest Classifier : Validation Set ',rfc_val,'Test Set ',rfc_test)
print('Extra Forest Classifier : Validation Set ',etc_val,'Test Set ',etc_test)
print('Support Vector Classifier : Validation Set ',svc_val,'Test Set ',svc_test)
print('Soft Voting Classifier : Validation Set ',soft_vc_val,'Test Set ',soft_vc_test)