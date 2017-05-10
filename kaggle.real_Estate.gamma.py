######################################################################### Set Variables #####################################################################
#Regularization alphas for our Linear functions
alphas = [.00001,.0001,.001,.01,.1,.5,1,5,10,100,1000]

#KFold for Cross Validation functions
kfold = 20

#Skewness threshold to take log of very skewed features
SKEW = .75

#Size of training Set
TRAIN_ROWS = 1461
TRAIN_COLS = 2920

#XGB Learning Rate
L_RATE = .01

#XGB Boosting Parameters
param = {'max_depth':7, 'eta':.75, 'silent':1, 'objective':'reg:linear' }

#Number of Boosing rounds
n_rounds = 10

#Number of NFolds
n_folds = 5

#Metric to be used for XGBoost
MET = 'rmse'

######################################################################## Import Data ######################################################
import numpy as np
import pandas as pd
test_x = pd.read_csv('test.csv')
train_x = pd.read_csv('train.csv')
train_x = train_x.loc[train_x.GrLivArea < 4000] #Rid of very large properties
y = (train_x['SalePrice']) 
train_x = train_x.drop('SalePrice',axis=1) 
all_data = pd.concat((train_x.loc[:,'MSSubClass':'SaleCondition'],test_x.loc[:,'MSSubClass':'SaleCondition']))



######################################################################## graphing sales price skew ######################################################
import matplotlib.pyplot as plt

#Create Axis
plt.figure(1)

#Plot unaltered y
plt.subplot(121)
plt.hist(y)
plt.ylabel('Count')
plt.xlabel('SalesPrice')
plt.title('Histogram of SalesPrice')

#Alter train_y by taking log
y = np.log1p(y)

#plot altered y
plt.subplot(122)
plt.hist(y)
plt.ylabel('Count')
plt.xlabel('SalesPrice')
plt.title('Histogram of log(SalesPrice + 1)')
# plt.show()




####################################################################### Preprocessing the data #######################################################################

#Obtain Continous features
is_Cont = all_data.dtypes[all_data.dtypes != "object"].index

#Obtain Categorical features
is_Cat = all_data.dtypes[all_data.dtypes == "object"].index

#Used to take skewness of features
from scipy.stats import skew

#Obtain measures of skewness of each continous feature
skewed_feats = all_data[is_Cont].apply(lambda x: skew(x.dropna()))

#Only keep values that are above a certain level of skewness
skewed_feats = skewed_feats[skewed_feats > SKEW]

#Obtain Index values
skewed_feats = skewed_feats.index

#Take logs of skewed features
all_data[skewed_feats] = np.log1p(all_data[skewed_feats])

#Turn categorical features into a collection of binary features
all_data = pd.get_dummies(all_data)

#Fill NA values with the mean
all_data = all_data.fillna(all_data.mean())

#Create your training and testing data
test_x = all_data[train_x.shape[0]:]

train_x = all_data[:train_x.shape[0]]



####################################################################### modeling #######################################################################


#Import Linear Modeling Modules to run models
import time
from sklearn import svm,datasets,linear_model, preprocessing, tree
from sklearn.linear_model import LassoCV, ElasticNetCV, RidgeCV
from sklearn.model_selection import *	

#Use Simple Lasso to Cross Validate
t1 = time.time()
model_lasso = LassoCV(alphas=alphas,random_state=0).fit(train_x,y)
t_lasso = time.time() - t1
print("\nLasso Score with CV: "+str(-1*cross_val_score(model_lasso,train_x,y,scoring='neg_mean_squared_error',cv=kfold).mean())+
	  "\nTime of "+str(t_lasso))
pred_lasso = pd.DataFrame(data=np.expm1(model_lasso.predict(test_x)),    # values
              			  index= range(TRAIN_ROWS,TRAIN_COLS), #Set Index
              			  columns=['SalePrice'])   # 1st column as index

#Use Elastic Net to Cross Validate
t2 = time.time()
model_elastic = ElasticNetCV(alphas=alphas).fit(train_x,y)
t_elastic = time.time() - t1
print("\nElastic Net Score with CV: "+str(-1*cross_val_score(model_elastic,train_x,y,scoring='neg_mean_squared_error',cv=kfold).mean())+
	  "\nTime of "+str(t_elastic))
pred_elastic = pd.DataFrame(data=np.expm1(model_elastic.predict(test_x)),    # values
              			  index= range(TRAIN_ROWS,TRAIN_COLS), #Set Index
              			  columns=['SalePrice'])   # 1st column as index

#Use Ridge to Cross Validate and Model
t3 = time.time()
model_ridge = RidgeCV(alphas=alphas).fit(train_x,y)
t_ridge = time.time() - t1
print("\nRidge Score with CV: "+str(-1*cross_val_score(model_ridge,train_x,y,scoring='neg_mean_squared_error',cv=kfold).mean())+
	  "\nTime of "+str(t_ridge))
pred_ridge = pd.DataFrame(data=np.expm1(model_ridge.predict(test_x)),    # values
              			  index= range(TRAIN_ROWS,TRAIN_COLS), #Set Index
              			  columns=['SalePrice'])   # 1st column as index

#Use Random Forest to make estimator
from sklearn.ensemble import RandomForestRegressor
t4 = time.time()
clf = RandomForestRegressor().fit(train_x, np.asarray(y, dtype="|S6"))
t_rf = time.time() - t4
print("\nRandom Forest Score with CV: "+str(np.mean(cross_val_score(clf, train_x, y, cv=10)))+
	  "\nTime of "+str(t_rf))
pred_rf = pd.DataFrame(data=np.expm1(clf.predict(test_x)),    # values
              			  index= range(TRAIN_ROWS,TRAIN_COLS), #Set Index
              			  columns=['SalePrice'])   # 1st column as index

#Import XGBosst and use CV to obtain estimator 
import xgboost as xgb
t5 = time.time()
cv = xgb.cv(param,xgb.DMatrix(data=train_x, label = y), n_rounds,nfold = n_folds, metrics = MET)
model_xgb = xgb.XGBRegressor(max_depth=5,
							 learning_rate=1,
							 silent=1, 
							 objective='reg:linear').fit(train_x,y)
t_xgb = time.time() - t5
print("\n Extreme Gradient Boosting Score with CV: "+str(cv.mean())+
	  "\nTime of "+str(t_xgb))
pred_xgb = pd.DataFrame(data=np.expm1(model_xgb.predict(test_x)),    # values
              			index= range(TRAIN_ROWS,TRAIN_COLS), #Set Index
              			columns=['SalePrice'])   # 1st column as index











all_lasso = (pred_lasso + pred_elastic + pred_ridge + pred_xgb + pred_rf)/5
all_lasso.to_csv('output.csv', header=True, index_label='Id')



# ####################################################################### Most and least influential variables #######################################################################









