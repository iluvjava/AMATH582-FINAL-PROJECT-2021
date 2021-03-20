#!/usr/bin/env python
# coding: utf-8

# In[1]:


from emnist import extract_training_samples, extract_test_samples
import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
from keras.layers import *
from keras.models import *
from keras.optimizers import *
from keras.utils import to_categorical
import kerastuner as kt
import pandas as pd
from keras.models import load_model

from core import *

# returns a compiled model
# identical to the previous one
model = load_model('./CNN_Models/custom-model-digits-best.h5')


# In[2]:


test_X, test_y = extract_test_samples("digits");
test_X = test_X.astype('float32')
test_X /= 255
test_y = to_categorical(test_y)
test_X = test_X.reshape(-1, 28, 28, 1)


# In[ ]:


model.summary()


# In[3]:


result = model.evaluate(test_X, test_y)
print("[test loss, test accuracy]:", result)


# In[4]:


y_pred = model.predict(test_X)
y_pred_label = np.argmax(y_pred,axis=1)
# y_pred_label = np.where(y_pred_label==1)[1]

_, test_y_label = extract_test_samples("digits");


# In[7]:


Conmat = ConfusionMatrix(test_y_label, y_pred_label)
Conmat.visualize()
Conmat.report()


# In[5]:


df=pd.read_csv('./CNN_Models/history.csv')


# In[6]:


loss = df['loss'].to_numpy()
val_loss = df['val_loss'].to_numpy()
accuracy = df['accuracy'].to_numpy()
val_accuracy = df['val_accuracy'].to_numpy()


# In[77]:


# plot loss during training
plt.subplot(211)
plt.title('Loss')
plt.plot(loss, label='train')
plt.plot(val_loss, label='val')
plt.xlabel('epoch')
plt.legend()
# plot accuracy during training
plt.subplot(212)
plt.title('Accuracy')
plt.plot(accuracy, label='train')
plt.plot(val_accuracy, label='val')
plt.xlabel('epoch')
plt.legend()

plt.tight_layout()
plt.savefig('CNN-digits-best.pdf')
plt.show()


# In[ ]:




