#!/usr/bin/env python
# coding: utf-8

# # Diabetes dataset prediction by KODI VENU

# In[1]:


import pandas as pd
data=pd.read_csv('diabetes.csv')


# Top 5 rows

# In[2]:


data.head()


# Last 5 rows

# In[4]:


data.tail()


# Dataset Shape

# In[5]:


data.shape


# In[6]:


print('Number of rows', data.shape[0])
print('Number of columns', data.shape[1])


# Dataset Information

# In[7]:


data.info()


# Check null values in the dataset

# In[8]:


data.isnull().sum()


# Dataset Statistics

# In[9]:


data.describe()


# In[10]:


import numpy as np
data_copy=data.copy(deep=True)
data.columns


# In[13]:


data_copy[['Glucose','BloodPressure','SkinThickness','Insulin','BMI']]=data_copy[['Glucose','BloodPressure','SkinThickness','Insulin','BMI']].replace(0,np.nan)


# In[14]:


data_copy.isnull().sum()


# In[15]:


data['Glucose']=data['Glucose'].replace(0,data['Glucose'].mean())


# In[16]:


data['BloodPressure']=data['BloodPressure'].replace(0,data['BloodPressure'].mean())


# In[17]:


data['SkinThickness']=data['SkinThickness'].replace(0,data['SkinThickness'].mean())


# In[18]:


data['Insulin']=data['Insulin'].replace(0,data['Insulin'].mean())


# In[19]:


data['BMI']=data['BMI'].replace(0,data['BMI'].mean())


# Store feature matrix in X and target in y

# In[21]:


X= data.drop('Outcome',axis=1)
y=data['Outcome']


# Train/Test split

# In[22]:


from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.20,random_state=42)


# Scikit Learn Pipeline

# In[25]:


from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.pipeline import Pipeline


# In[28]:


pipeline_lr=Pipeline([('scalar1',StandardScaler()),('lr_classifier',LogisticRegression())])


# In[29]:


pipeline_knn=Pipeline([('scalar2',StandardScaler()),('knn_classifier',KNeighborsClassifier())])
pipeline_svc=Pipeline([('scalar3',StandardScaler()),('SVC_classifier',SVC())])


# In[31]:


pipeline_dt=Pipeline([('dt_classifier',DecisionTreeClassifier())])
pipeline_rf=Pipeline([('rf_classifier',RandomForestClassifier())])
pipeline_gbc=Pipeline([('gbc_classifier',GradientBoostingClassifier())])


# In[33]:


pipelines=[pipeline_lr,pipeline_knn,pipeline_svc,pipeline_dt,pipeline_rf,pipeline_gbc]


# In[34]:


pipelines


# In[35]:


for pipe in pipelines:
    pipe.fit(X_train,y_train)


# In[36]:


pipe_dict={0:'LR',1:'KNN',2:'SVC',3:'DT',4:'RF',5:'GBC'}
pipe_dict


# In[37]:


for i, model in enumerate(pipelines):
    print("{} Test Accuracy:{}".format(pipe_dict[i],model.score(X_test,y_test))*100)


# In[38]:


from sklearn.ensemble import RandomForestClassifier


# In[40]:


X=data.drop('Outcome',axis=1)
y=data['Outcome']


# In[41]:


rf=RandomForestClassifier()
rf=RandomForestClassifier(max_depth=3)


# In[42]:


rf.fit(X,y)


# Prediction on New Data

# In[46]:


new_data=pd.DataFrame({'Pregnancies':6,'Glucose':148.0,'BloodPressure':72.0,'SkinThickness':35.0,'Insulin':9.799479,'BMI':33.6,'DiabetesPedigreeFunction':0.627,'Age':50},index=[0])


# In[48]:


p=rf.predict(new_data)


# In[49]:


if p[0]==0:
    print('non_diabetic')
else:
    print('diabetic')


# Save model using joblib

# In[50]:


import joblib
joblib.dump(rf,'model_joblib_diabetes')


# In[51]:


model=joblib.load('model_joblib_diabetes')
model.predict(new_data)


# In[ ]:




