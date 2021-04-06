from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.svm import SVC
from sklearn.ensemble import VotingClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import PredefinedSplit
from sklearn.svm import LinearSVC
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler


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

#Performing Grid Search to optimize the hyperparameter
rfc=RandomForestClassifier(n_jobs=-1,warm_start=True)
n_est_linspace=11
max_depth_linspace=6
param_grid_rfc={'n_estimators':np.linspace(1,1000,n_est_linspace).astype(int),'max_depth':np.linspace(1,25,max_depth_linspace).astype(np.int)}
gscv_rfc=GridSearchCV(estimator=rfc,param_grid=param_grid_rfc,scoring="accuracy",n_jobs=-1,verbose=10,cv=ps)
gscv_rfc.fit(x_train_val,y_train_val)
print(gscv_rfc.best_score_)
print(gscv_rfc.best_params_)

#Plotting the results
scores=[]
for score in gscv_rfc.cv_results_["split0_test_score"]:
    scores=np.append(scores,score)
x_estimators=[]
for estimators in gscv_rfc.cv_results_["param_n_estimators"]:
    x_estimators=np.append(x_estimators,estimators)
y_depths=[]
for depth in gscv_rfc.cv_results_["param_max_depth"]:
    y_depths=np.append(y_depths,depth)

x_estimators_graph=x_estimators.reshape(max_depth_linspace,n_est_linspace)
scores_graph_x=scores.reshape(max_depth_linspace,n_est_linspace)
scores_graph_y=scores.reshape(n_est_linspace,max_depth_linspace,order='F')
y_depths_graph=y_depths.reshape(n_est_linspace,max_depth_linspace,order='F')  

for i in range(max_depth_linspace):
    
    plt.plot(x_estimators_graph[i,:],scores_graph_x[i,:],label="max_depth = "+y_depths_graph[0,i].astype(str))
    plt.legend()
    plt.title("Random Forest Score vs # of Estimators")
    plt.rcParams["figure.figsize"] = (10,10)
    plt.savefig('rf_Score_Estimators.png', bbox_inches='tight')


for i in range(n_est_linspace):
    
    plt.plot(y_depths_graph[i,:],scores_graph_y[i,:],label="n_estimator = "+x_estimators_graph[0,i].astype(str))
    plt.legend()
    plt.title("Random Forest Score vs Maximum Depth")
    plt.rcParams["figure.figsize"] = (10,10)
    plt.savefig('rf_Score_MaxDepth.png', bbox_inches='tight')
   

#Saving the results
data=gscv_rfc.cv_results_
np.save("rf_data.npy",data)

#Best Hyperparameters
#{'max_depth': 15, 'n_estimators': 200}
