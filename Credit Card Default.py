#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')
from sklearn.feature_selection import RFE
from sklearn.model_selection import KFold # for cross validation
from sklearn.model_selection import GridSearchCV # for tuning parameter
from sklearn.model_selection import RandomizedSearchCV  # Randomized search on hyper parameters.
from sklearn.preprocessing import StandardScaler # for normalization
from sklearn.pipeline import Pipeline 
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.neighbors import KNeighborsClassifier #KNN


# In[2]:


data=pd.read_csv("C:/Users/abcdr/OneDrive/Desktop/UCI_Credit_Card.csv")


# In[4]:


data.head()
data.drop('ID', axis = 1, inplace =True)


# SEX: Gender
#               1 = male 
#               2 = female
#               
#               
# EDUCATION:
#                1 = graduate school 
#                2 = university 
#                3 = high school 
#                4 = others 
#                5 = unknown 
#                6 = unknown
#                
#                
#                
# PAY_0,2,3,4,5,6: 
#                 Repayment status in September 2005, August 2005, July 2005, June 2005, May 2005, April 2005 (respectivey)
#               -2= no consumption
#               -1= pay duly
#               1 = payment delay for one month
#               2 = payment delay for two months
#               ... 
#               8 = payment delay for eight months
#               9 = payment delay for nine months and above

# In[5]:


data.rename(columns={'default.payment.next.month':'Default'},inplace=True)


# In[6]:


data['EDUCATION'].unique()


# In[7]:


data['EDUCATION']=np.where(data['EDUCATION'] == 5, 4, data['EDUCATION'])
data['EDUCATION']=np.where(data['EDUCATION'] == 6, 4, data['EDUCATION'])
data['EDUCATION']=np.where(data['EDUCATION'] == 0, 4, data['EDUCATION'])


# In[8]:


data['EDUCATION'].unique()


# In[9]:


data['MARRIAGE'].unique()


# In[10]:


sns.countplot(data['Default'].value_counts())
plt.title('COUNT OF CREDIT CARDS', size=14)
data['Default'].value_counts()


# In[11]:


print("SUMMARY STATISTICS OF NUMERIC COLUMNS")
print(data.describe().T)


# In[12]:


pay=['PAY_0','PAY_2','PAY_3','PAY_4','PAY_5','PAY_6']


# In[13]:


data[pay].hist(layout=(2,3))


# In[14]:


pay_amt=['PAY_AMT1','PAY_AMT2','PAY_AMT3','PAY_AMT4','PAY_AMT5','PAY_AMT6']


# In[15]:


data[pay_amt].hist(layout=(2,4))


# In[16]:


corr = data.corr()


# In[17]:


f,ax = plt.subplots(figsize=(10, 7))
sns.heatmap(corr, cbar = True,  square = True, annot = False, fmt= '.1f', 
            xticklabels= True, yticklabels= True
            ,cmap="coolwarm", linewidths=.5,ax=ax)
plt.title('CORRELATION MATRIX - HEATMAP', size=18)


# In[18]:


X = data.drop('Default', axis=1)  
Y= data['Default']
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression


# In[19]:


X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size=0.2, random_state=101)


# In[20]:


num_features=4
model = LogisticRegression()
rfe_stand = RFE(model,num_features)
fit_stand = rfe_stand.fit(X,Y)
print("Std Model Feature Ranking:", fit_stand.ranking_)
score_stand = rfe_stand.score(X,Y)
print("Standardized Model Score with selected features is: %f" % (score_stand.mean()))


# In[21]:


print("St Model Num Features:", fit_stand.n_features_)
print("St Model Selected Features:", fit_stand.support_)


# In[21]:


feature_names = np.array(X.columns)
print('Most important features (RFE): %s'% feature_names[rfe_stand.support_])


# In[22]:


X_new=data[['SEX','MARRIAGE','PAY_0','PAY_3']]


# In[24]:


X_new_train,X_new_test,Y_train,Y_test=train_test_split(X_new,Y,test_size=0.35,random_state=101)


# In[28]:


glm=LogisticRegression()


# In[29]:


glm.fit(X_new_train,Y_train)


# In[30]:


y_pred=glm.predict(X_new_test)


# In[31]:


from sklearn.metrics import classification_report


# In[32]:


report=classification_report(Y_test,y_pred)


# In[33]:


print(report)


# In[34]:


from sklearn.model_selection import cross_val_score,ShuffleSplit


# In[35]:


cv=ShuffleSplit(n_splits=7,test_size=0.35,random_state=101)


# In[37]:


cross_val_score(glm,X_new,Y,cv=cv).mean()


# In[38]:


param_dist = {"max_depth": [1,2,3,4,5,6,7,8,9],
              "max_features": [1,2,3,4,5,6,7,8,9],
              "min_samples_leaf": [1,2,3,4,5,6,7,8,9],
              "criterion": ["gini", "entropy"]}


# In[40]:


from sklearn.tree import DecisionTreeClassifier
tree = DecisionTreeClassifier()


# In[41]:


tree_cv = RandomizedSearchCV(tree, param_distributions=param_dist, cv=5, random_state=0)


# In[45]:


tree_cv.fit(X_new_train,Y_train)
print("Tuned Decision Tree Parameters: {}".format(tree_cv.best_params_))


# In[51]:


Tree = DecisionTreeClassifier(criterion= 'gini', max_depth= 7, 
                                     max_features= 4, min_samples_leaf= 2, 
                                     random_state=0)


# In[52]:


Tree.fit(X_new_train,Y_train)


# In[53]:


y_pred_tree=Tree.predict(X_new_test)


# In[54]:


report_tree=classification_report(Y_test,y_pred_tree)


# In[55]:


print(report_tree)


# In[56]:


cv=ShuffleSplit(n_splits=7,test_size=0.35,random_state=101)


# In[58]:


cross_val_score(Tree,X_new,Y,cv=cv).mean()


# In[59]:


param_dist={'n_estimators': [50,100,150,200,250],
               "max_features": [1,2,3,4,5,6,7,8,9],
               'max_depth': [1,2,3,4,5,6,7,8,9],
               "criterion": ["gini", "entropy"]}
    


# In[60]:


rf=RandomForestClassifier()


# In[61]:


rf_cv=RandomizedSearchCV(rf,param_distributions=param_dist,cv=5)


# In[64]:


rf_cv.fit(X_new,Y)


# In[65]:


print("Tuned Random Forest Parameters: %s" % (rf_cv.best_params_))


# In[66]:


Ran = RandomForestClassifier(criterion= 'gini', max_depth= 1, 
                                     max_features= 4, n_estimators= 250, 
                                     random_state=0)


# In[67]:


Ran.fit(X_new_train,Y_train)


# In[68]:


y_pred_forest=Ran.predict(X_new_test)


# In[69]:


report_forest=classification_report(Y_test,y_pred_forest)


# In[70]:


print(report_forest)


# In[71]:


from sklearn.metrics import roc_curve,roc_auc_score


# In[72]:


fpr1,tpr1,_=roc_curve(Y_test,y_pred)
auc1=roc_auc_score(Y_test,y_pred)


# In[73]:


fpr2,tpr2,_=roc_curve(Y_test,y_pred_tree)
auc2=roc_auc_score(Y_test,y_pred_tree)


# In[74]:


fpr3,tpr3,_=roc_curve(Y_test,y_pred_forest)
auc3=roc_auc_score(Y_test,y_pred_forest)


# In[78]:


plt.plot([0, 1], [0, 1], 'k--')
plt.plot(fpr1,tpr1,label="Logistic Regression, auc="+str(round(auc1,2)))
plt.plot(fpr2,tpr2,label="Decision Tree, auc="+str(round(auc2,2)))
plt.plot(fpr3,tpr3,label="Random Forest, auc="+str(round(auc3,2)))
plt.legend(loc=4, title='Models', facecolor='white')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC', size=15)
plt.box(False)


# In[3]:


data.head()


# In[22]:


import pickle
filename = 'credit.pkl'
pickle.dump(model, open(filename, 'wb'))


# In[ ]:




