#!/usr/bin/env python
# coding: utf-8

# ### Importing Libraries

# In[2]:


import pandas as pd
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.metrics import recall_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.tree import DecisionTreeClassifier
from imblearn.combine import SMOTEENN


# #### Reading csv

# In[2]:


df=pd.read_csv("tel_churn.csv")
df.head()


# In[3]:


df=df.drop('Unnamed: 0',axis=1)


# In[4]:


x=df.drop('Churn',axis=1)
x


# In[5]:


y=df['Churn']
y


# ##### Train Test Split

# In[6]:


x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2)


# #### Decision Tree Classifier

# In[7]:


model_dt=DecisionTreeClassifier(criterion = "gini",random_state = 100,max_depth=6, min_samples_leaf=8)


# In[8]:


model_dt.fit(x_train,y_train)


# In[9]:


y_pred=model_dt.predict(x_test)
y_pred


# In[10]:


model_dt.score(x_test,y_test)


# In[11]:


print(classification_report(y_test, y_pred, labels=[0,1]))


# ###### As you can see that the accuracy is quite low, and as it's an imbalanced dataset, we shouldn't consider Accuracy as our metrics to measure the model, as Accuracy is cursed in imbalanced datasets.
# 
# ###### Hence, we need to check recall, precision & f1 score for the minority class, and it's quite evident that the precision, recall & f1 score is too low for Class 1, i.e. churned customers.
# 
# ###### Hence, moving ahead to call SMOTEENN (UpSampling + ENN)

# In[12]:


sm = SMOTEENN()
X_resampled, y_resampled = sm.fit_sample(x,y)


# In[13]:


xr_train,xr_test,yr_train,yr_test=train_test_split(X_resampled, y_resampled,test_size=0.2)


# In[14]:


model_dt_smote=DecisionTreeClassifier(criterion = "gini",random_state = 100,max_depth=6, min_samples_leaf=8)


# In[15]:


model_dt_smote.fit(xr_train,yr_train)
yr_predict = model_dt_smote.predict(xr_test)
model_score_r = model_dt_smote.score(xr_test, yr_test)
print(model_score_r)
print(metrics.classification_report(yr_test, yr_predict))


# In[16]:


print(metrics.confusion_matrix(yr_test, yr_predict))


# ###### Now we can see quite better results, i.e. Accuracy: 92 %, and a very good recall, precision & f1 score for minority class.
# 
# ###### Let's try with some other classifier.

# #### Random Forest Classifier

# In[17]:


from sklearn.ensemble import RandomForestClassifier


# In[18]:


model_rf=RandomForestClassifier(n_estimators=100, criterion='gini', random_state = 100,max_depth=6, min_samples_leaf=8)


# In[19]:


model_rf.fit(x_train,y_train)


# In[20]:


y_pred=model_rf.predict(x_test)


# In[21]:


model_rf.score(x_test,y_test)


# In[22]:


print(classification_report(y_test, y_pred, labels=[0,1]))


# In[23]:


sm = SMOTEENN()
X_resampled1, y_resampled1 = sm.fit_sample(x,y)


# In[24]:


xr_train1,xr_test1,yr_train1,yr_test1=train_test_split(X_resampled1, y_resampled1,test_size=0.2)


# In[25]:


model_rf_smote=RandomForestClassifier(n_estimators=100, criterion='gini', random_state = 100,max_depth=6, min_samples_leaf=8)


# In[26]:


model_rf_smote.fit(xr_train1,yr_train1)


# In[27]:


yr_predict1 = model_rf_smote.predict(xr_test1)


# In[28]:


model_score_r1 = model_rf_smote.score(xr_test1, yr_test1)


# In[29]:


print(model_score_r1)
print(metrics.classification_report(yr_test1, yr_predict1))


# In[30]:


print(metrics.confusion_matrix(yr_test1, yr_predict1))


# #### Performing PCA

# In[31]:


# Applying PCA
from sklearn.decomposition import PCA
pca = PCA(0.9)
xr_train_pca = pca.fit_transform(xr_train1)
xr_test_pca = pca.transform(xr_test1)
explained_variance = pca.explained_variance_ratio_


# In[32]:


model=RandomForestClassifier(n_estimators=100, criterion='gini', random_state = 100,max_depth=6, min_samples_leaf=8)


# In[33]:


model.fit(xr_train_pca,yr_train1)


# In[34]:


yr_predict_pca = model.predict(xr_test_pca)


# In[35]:


model_score_r_pca = model.score(xr_test_pca, yr_test1)


# In[36]:


print(model_score_r_pca)
print(metrics.classification_report(yr_test1, yr_predict_pca))


# #### Pickling the model

# In[37]:


import pickle


# In[38]:


filename = 'model.sav'


# In[39]:


pickle.dump(model_rf_smote, open(filename, 'wb'))


# In[40]:


load_model = pickle.load(open(filename, 'rb'))


# In[41]:


model_score_r1 = load_model.score(xr_test1, yr_test1)


# In[42]:


model_score_r1

