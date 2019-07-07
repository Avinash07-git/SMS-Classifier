#!/usr/bin/env python
# coding: utf-8

# In[8]:


import pandas as pd
import numpy as np

df = pd.read_table(r'H:\C.V\Github Kaggle\6. Spam Mails Classifier\SMSSpamCollection+(1)', header = None, names = ('class', 'sms'))
df.head()


# In[11]:


# counting spam and ham instances
# df.column_name.value_counts() - gives no. of unique inputs in the columns

spam_ham = df['class'].value_counts()
spam_ham


# In[16]:


# mapping labels to 0 and 1
df['label'] = df['class'].map({'ham':0, 'spam':1})
df.head()


# In[17]:


X = df.sms
y = df.label


# In[19]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split (X,y,random_state = 1)


# In[22]:


# vectorizing the sentences; removing stop words
from sklearn.feature_extraction.text import CountVectorizer
vect = CountVectorizer (stop_words = 'english')


# In[23]:


vect.fit(X_train)


# In[25]:


vect.vocabulary_


# In[26]:


# transforming the train and test datasets
X_train_transformed = vect.transform(X_train)
X_test_transformed =vect.transform(X_test)


# In[27]:


# note that the type is transformed matrix
print(type(X_train_transformed))
print(X_train_transformed)


# In[34]:


# training the NB model and making predictions

from sklearn.naive_bayes import MultinomialNB
mnb = MultinomialNB()

#fit
mnb.fit (X_train_transformed, y_train)

#predict class
y_pred_class = mnb.predict (X_test_transformed)


#predict probablities
Y_pred_proba = mnb.predict_proba (X_test_transformed)

# printing the overall accuracy
from sklearn import metrics
metrics.accuracy_score(y_test, y_pred_class)


# In[35]:


y_pred_class


# In[36]:


mnb


# In[37]:


metrics.confusion_matrix (y_test, y_pred_class)


# In[38]:


confusion = metrics.confusion_matrix(y_test, y_pred_class)
print(confusion)
#[row, column]
TN = confusion[0, 0]
FP = confusion[0, 1]
FN = confusion[1, 0]
TP = confusion[1, 1]


# In[39]:


sensitivity = TP / float(FN + TP)
print("sensitivity",sensitivity)


# In[40]:


specificity = TN / float(TN + FP)

print("specificity",specificity)


# In[41]:


precision = TP / float(TP + FP)

print("precision",precision)
print(metrics.precision_score(y_test, y_pred_class))


# In[43]:


Y_pred_proba


# In[44]:


print("precision",precision)
print("PRECISION SCORE :",metrics.precision_score(y_test, y_pred_class))
print("RECALL SCORE :", metrics.recall_score(y_test, y_pred_class))
print("F1 SCORE :",metrics.f1_score(y_test, y_pred_class))


# In[46]:


# creating an ROC curve
from sklearn.metrics import confusion_matrix as sk_confusion_matrix
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

false_positive_rate, true_positive_rate, thresholds = roc_curve(y_test, Y_pred_proba[:,1])
roc_auc = auc(false_positive_rate, true_positive_rate)


# In[47]:


# area under the curve
print (roc_auc)


# In[48]:


plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.title('ROC')
plt.plot(false_positive_rate, true_positive_rate)


# In[ ]:




