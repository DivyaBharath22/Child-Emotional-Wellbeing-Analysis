# -*- coding: utf-8 -*-
"""
Created on Fri Apr  1 17:56:04 2022

@author: Admin
"""

############ Import Libraries #################
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore",category=UserWarning)

########## Read Data into python ##################
emotional = pd.read_csv("C:/Users/Admin/Desktop/Project/Emotional_data.csv")
type(emotional)
emotional.info()
emotional.shape 

########## Drop Unnamed column ########
emotional.drop(['Unnamed: 0'], axis=1, inplace = True)

###### EDA ######
emotional.describe()

# Rearranging the columns
emotional = emotional.iloc[:,[38,0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36]]

# Separating input and output variables
X = emotional.iloc[:,1:38]
y = emotional.iloc[:,0]

X.shape
y.shape

###### Score and P Value ########
# Lets use the sklearn chi2 function
from sklearn.feature_selection import chi2,SelectKBest
cs = SelectKBest(score_func=chi2,k=37)
cs.fit(X,y)
feature_score_pvalue = pd.DataFrame({"Score":cs.scores_, "P_value":np.round(cs.pvalues_,3)}, index=X.columns)
feature_score_pvalue.nlargest(n=37,columns="Score")

########## Splitting into train and test data #########
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,random_state = 42,test_size =0.30)
print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)

########### Multinomial Logistic Regression without feature selection ###############

# ‘multinomial’ option is supported only by the ‘lbfgs’ and ‘newton-cg’ solvers
# Creating the prediction model
from sklearn.linear_model import LogisticRegressionCV
multi_model = LogisticRegressionCV(multi_class = "multinomial", solver = "newton-cg").fit(X_train,y_train)

# Evaluate the metrics
from sklearn.metrics import classification_report,confusion_matrix,roc_curve,auc,accuracy_score
# Test data prediction 
pred_test_multi = multi_model.predict(X_test)
print("Accuracy_score:",accuracy_score(y_test,pred_test_multi))

# Train data prediction
pred_train_multi = multi_model.predict(X_train)
print("Accuracy_score:",accuracy_score(y_train,pred_train_multi))

################# Random Forest Without feature selection ##############
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFE,SelectFromModel

rf_model = RandomForestClassifier(n_estimators=100,random_state=42).fit(X_train,y_train)

# Test data prediction
pred_test_rf = rf_model.predict(X_test)
print("Accuracy_score:",accuracy_score(y_test,pred_test_rf))

# Train data prediction
pred_train_rf = rf_model.predict(X_train)
print("Accuracy_score:",accuracy_score(y_train,pred_train_rf))

###### Feature Selection using Random Forest ###########
# Get the importance of the resulting features
important_values = rf_model.feature_importances_
print(important_values)

# Create a dataframe for visualization
features_emotional = pd.DataFrame({"Features":X_train.columns,"Score":important_values} )
features_emotional.set_index('Features')

# Sort in ascending order for better visualization
features_emotional = features_emotional.sort_values('Score')

# Plot the feature importances in bars
plt.figure(figsize=(10,3))
plt.xticks(rotation=45)
sns.barplot(x='Features',y='Score',data=features_emotional)

# Choosing less important and more important columns based on the score
features_emotional['Score'] = np.where(features_emotional['Score']<=0.025,'less Important','more important')

# Retieving most Important Features
# Creating a dataframe with the selected features
df = pd.DataFrame(emotional.iloc[:,[0,9,11,12,16,28,29,30,31,32,33]])
X = df.iloc[:,1:11]
y = df.iloc[:,0]

# Check for the Correlation
df.corr()

# Plot the heatmap between features and Target
corr = df.corr(method ='pearson')
plt.figure(figsize=(8,4))
sns.heatmap(corr,cmap = 'RdYlGn',annot=True)

# Plot the heatmap between features
corr = df.iloc[:,1:11].corr(method ='pearson')
plt.figure(figsize=(8,4))
sns.heatmap(corr,cmap = 'RdYlGn',annot=True)

# Scatter plot between each possible pair of independent variable and also histogram for each independent variable 
sns.pairplot(df)

# Splitting into train and test data
X_train,X_test,y_train,y_test = train_test_split(X,y,random_state = 42,test_size =0.30)

########### Multinomial Logistic Regression with feature selection ###############

# Creating the prediction model
multi_model_fs = LogisticRegressionCV(multi_class = "multinomial", solver = "newton-cg").fit(X_train,y_train)

# Test Accuracy and Classification report
pred_test_multi_fs = multi_model_fs.predict(X_test)
print("Accuracy_score:",accuracy_score(y_test,pred_test_multi_fs))

# Train Accuracy and Classification report
pred_train_multi_fs = multi_model_fs.predict(X_train)
print("Accuracy_score:",accuracy_score(y_train,pred_train_multi_fs))

############### GridSearch CV ######################
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
kfold = KFold(n_splits=5, random_state=42, shuffle=True)
rf_clf_grid = RandomForestClassifier(n_estimators=500, n_jobs=-1, random_state=42)
param_grid = {"max_features": [4,5,6,7,8,9,10], "min_samples_split":[2, 3, 10]}
grid_search = GridSearchCV(rf_clf_grid, param_grid, n_jobs = -1, cv =kfold, scoring = 'accuracy').fit(X, y)
grid_search.best_params_
cv_rf_clf_grid = grid_search.best_estimator_
results = cross_val_score(cv_rf_clf_grid, X, y, cv=kfold)
print('Accuracy_score:',results.mean())

################ f1 Score #############

from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn import metrics
from sklearn.metrics import precision_score, recall_score, f1_score

# Standardize the data
sc = StandardScaler()
sc.fit(X_train)
X_train_std = sc.transform(X_train)
X_test_std = sc.transform(X_test)
svc = SVC(kernel ='linear', C = 10.0, random_state = 42)
svc.fit(X_train, y_train)

y_pred_svc = svc.predict(X_test)
confusion_matrix = confusion_matrix(y_true=y_test, y_pred=y_pred_svc)
print(confusion_matrix)
classification_report = metrics.classification_report(y_true=y_test, y_pred=y_pred_svc)
print(classification_report)

# Visualization
import matplotlib.cm as cm
fig, b = plt.subplots(figsize=(5, 15))
b.matshow(confusion_matrix, cmap=plt.cm.rainbow, alpha=0.3)
for i in range(confusion_matrix.shape[0]):
    for j in range(confusion_matrix.shape[1]):
        b.text(x=j, y=i,s=confusion_matrix[i, j], va='center', ha='center', size='x-large' )

plt.xlabel('Predictions', fontsize=15)
plt.ylabel('Actuals', fontsize=15)
plt.title('Confusion Matrix', fontsize=15)
plt.show()
    
################### SMOTE - Over Sampling Technique #######################
from imblearn.over_sampling import SMOTE
sm = SMOTE(random_state = 12)
X_train_res, y_train_res = sm.fit_resample(X_train, y_train)

# Applying Multinomial Logistic regeression
multi_model_rs = LogisticRegressionCV(multi_class = "multinomial", solver = "newton-cg").fit(X_train_res,y_train_res)
pred_multi_test_resampled = multi_model_rs.predict(X_test)
print("Accuracy_score:",accuracy_score(y_test,pred_multi_test_resampled))
pred_multi_train_resampled = multi_model_rs.predict(X_train)
print("Accuracy_score:",accuracy_score(y_train,pred_multi_train_resampled))

# Applying GridSearch CV
kfold = KFold(n_splits=5, random_state=42, shuffle=True)
rf_clf_grid = RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=42)
param_grid = {"max_features": [4,5,6,7,8,9,10], "min_samples_split":[2, 3, 10]}
grid_search_res = GridSearchCV(rf_clf_grid, param_grid, n_jobs = -1, cv =kfold, scoring = 'accuracy').fit(X_train_res, y_train_res)
grid_search_res.best_params_
cv_rf_clf_grid_res = grid_search_res.best_estimator_

y_test_pred_grid_res=cv_rf_clf_grid_res.predict(X_test)
print(accuracy_score(y_test,y_test_pred_grid_res))
pd.crosstab(pd.Series(y_test_pred_grid_res, name = 'Predicted'),pd.Series(y_test, name = 'Actual'))

y_train_pred_grid_res=cv_rf_clf_grid_res.predict(X_train)
print(accuracy_score(y_train,y_train_pred_grid_res))
pd.crosstab(pd.Series(y_train_pred_grid_res, name = 'Predicted'),pd.Series(y_train, name = 'Actual'))

# Deployment
import pickle

# Saving model to disk
pickle.dump(multi_model_fs,open('model1.pkl','wb'))
