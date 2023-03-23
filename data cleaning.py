#!/usr/bin/env python
# coding: utf-8

# # Categorical Data

# In[1]:


import pandas as pd


# In[2]:


salary_data=pd.DataFrame({'salary':[1900,3100,2500,5000],
                          'range':['low','mid','high','ultra']})


# In[3]:


salary_data


# In[4]:


a=salary_data.range.map({'low':1,'mid':2,'high':3,'ultra':4})


# In[5]:


a


# In[6]:


b=salary_data.salary.map({1900:10000,3100:20000,2500:30000,5000:40000})


# In[7]:


b


# # Normalization

# In[8]:


pip install matplotlib


# In[10]:


pip install scikit-learn


# In[14]:


import matplotlib.pyplot as plt


# In[15]:


from mpl_toolkits.mplot3d import Axes3D


# In[17]:


get_ipython().run_line_magic('matplotlib', 'inline')


# In[18]:


import numpy as np


# In[19]:


import pandas as pd


# In[21]:


normalization_df=pd.DataFrame({'x':np.random.randint(-100,100,1000),
                              'y':np.random.randint(-80,80,1000),
                              'z':np.random.randint(-150,150,1000)})


# In[23]:


normalization_df


# In[25]:


ax=plt.axes(projection='3d')

ax.scatter3D(normalization_df.x,normalization_df.y,
            normalization_df.z)
plt.figure()


# In[27]:


from sklearn.preprocessing import Normalizer


# In[28]:


normal=Normalizer()


# In[29]:


normalization_df=normal.fit_transform(normalization_df)


# In[30]:


normalization_df


# In[32]:


norm_df=pd.DataFrame(normalization_df,
                    columns=['x1','x2','x3'])


# In[33]:


norm_df


# In[35]:


ax=plt.axes(projection='3d')

ax.scatter3D(norm_df.x1,norm_df.x2,
            norm_df.x3)
plt.figure()


# # DATA SORTING

# In[38]:


data=[10,11,1,67,87,34,55,12,22,43,76,90,6,7,3,0,4,2,556,76,234]


# In[39]:


data.sort()
print(data)


# # MISSIND VALUES

# In[40]:


import numpy as np
import pandas as pd


# In[43]:


data={'First score':[np.nan,100,90,95],
     'Second scort':[30,np.nan,45,56],
     'Third scort':[40,80,98,np.nan]}


# In[45]:


scores=pd.DataFrame(data)


# In[46]:


scores


# In[47]:


scores.isnull()


# In[51]:


data={'First score':[np.nan,100,90,95],
     'Second scort':[30,np.nan,45,np.nan],
     'Third scort':[np.nan,80,np.nan,np.nan]}


# In[52]:


scores=pd.DataFrame(data)


# In[53]:


scores


# In[54]:


scores.isnull()


# In[59]:


from sklearn.impute import SimpleImputer


# In[61]:


imputer=SimpleImputer(missing_values=np.nan,
                    strategy='median')


# In[62]:


imputer=SimpleImputer(missing_values=np.nan,
                    strategy='mean')


# In[63]:


imputer=SimpleImputer(missing_values=np.nan,
                    strategy='mode')


# In[67]:


imputer.fit(scores)


# In[ ]:




