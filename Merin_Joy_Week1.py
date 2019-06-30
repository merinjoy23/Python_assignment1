#!/usr/bin/env python
# coding: utf-8

# In[85]:

# importing the pandas package
import pandas as pd


# In[86]:

# Loading the csv file as a dataframe
df = pd.read_csv('/Users/Merin/Documents/highUtilizationPredictionV2wco.csv')


# In[87]:

# Checking for any missing values or information in the dataframe
df
df.isnull().sum()
df.info()
df.size
len(df.index)


# In[92]:

# Storing the list of columns
df.columns
list(df.columns)
cls = df.columns.tolist()
cls


# In[96]:

# Removing unwanted columns
cls.remove('race')
cls.remove('patient_id')
cls.remove('HighUtilizationY2')
cls.remove('claimCount')
df[cls]


# In[98]:

# Spliting data into training and testing
sz = df.index.size
sz
tr = df[:int(sz*0.8)]
ts = df[int(sz*0.8):]
tr=tr.fillna(0)
ts=ts.fillna(0)


# In[102]:

# Selecting the input/output attributes
tr_inp=tr[cls]
tr_out=tr['HighUtilizationY2']
ts_inp=ts[cls]
ts_out=ts['HighUtilizationY2']


# In[103]:

# Importing LogisticRegression package
from sklearn.linear_model import LogisticRegression


# In[104]:

# Learn the LR model on train data
lr = LogisticRegression()
lr.fit(tr_inp, tr_out)


# In[110]:

# Importing RandomForestClassifier package
from sklearn.ensemble import RandomForestClassifier


# In[111]:

# Learn the RF model on train data
rf = RandomForestClassifier(n_estimators=100)
rf.fit(tr_inp, tr_out)


# In[113]:

# Import packages for AUC and ROC
from sklearn.metrics import auc
from sklearn.metrics import roc_curve


# In[121]:

# Apply LR on train and check AUC
probs=lr.predict_proba(tr_inp)[:,1]
fpr, tpr, thresholds = roc_curve(tr_out, probs)
auc(fpr,tpr) #0.8223835455617834


# In[122]:

# Apply LR on test and check AUC
probs=lr.predict_proba(ts_inp)[:,1]
fpr, tpr, thresholds = roc_curve(ts_out, probs)
auc(fpr,tpr) #0.8200381653450525


# In[123]:

# Apply RF on train and check AUC
rf_probs = rf.predict_proba(tr_inp)
rf_fpr, rf_tpr, rf_thresholds = roc_curve(tr_out, rf_probs[:,1])
auc(rf_fpr, rf_tpr) #0.9987994418715582


# In[124]:

# Apply RF on test and check AUC
rf_probs = rf.predict_proba(ts_inp)
rf_fpr, rf_tpr, rf_thresholds = roc_curve(ts_out, rf_probs[:,1])
auc(rf_fpr, rf_tpr) #0.7981046515794564



# In[178]:

# Load the 3 json files
r2 = pd.read_json('/Users/Merin/Documents/r2.json')
r3 = pd.read_json('/Users/Merin/Documents/r3.json')
r4 = pd.read_json('/Users/Merin/Documents/r4.json')


# In[179]:


# Predict on LR and RF models
r2.T
lr.predict_proba(r2.T[cls])[:,1] #array([1.00552705e-07]) - Not a high utilizer
rf.predict_proba(r2.T[cls])[:,1] #array([0.01]) - Not a high utilizer


# In[183]:


# r3
r3.T
# Add a new row for ELIX5 code with default value as '0'
r3.loc['ELIX5'] = 0
r3.T
# Replace 'Yes' to '1'
r3 = r3.replace('Yes',1)
# Predict on LR and RF models
lr.predict_proba(r3.T[cls])[:,1] #array([0.01503614]) - Not a high utilizer
rf.predict_proba(r3.T[cls])[:,1] #array([0.]) - Not a high utilizer


# In[195]:


# r4
r4.T
# Predict on LR and RF models
lr.predict_proba(r4.T[cls])[:,1] #array([0.07835953]) - Not a high utilizer
rf.predict_proba(r4.T[cls])[:,1] #array([0.11]) - Not a high utilizer
