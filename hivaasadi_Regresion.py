#!/usr/bin/env python
# coding: utf-8

# In[83]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics


# In[84]:


date=pd.read_csv("weatherAUS.csv")
date


# In[85]:


df=pd.DataFrame(date,columns=('Date','Location','MinTemp','MaxTemp','WindGustSpeed','Rainfall'))#,'WindSpeed9am','WindSpeed3pm','Humidity9am','Humidity3pm','Pressure9am','Pressure3pm','Temp9am','Temp3pm','Rainfall'))
df


# In[86]:


df_alb=df[df['Location']=='Albury']
df_alb


# In[87]:


df_alb.isnull().sum()


# In[88]:


df_alb.fillna(method="bfill",inplace=True)


# In[89]:


df_alb.isnull().sum()


# In[90]:


df_alb


# In[91]:


df_alb.dropna(inplace=True)
df_alb.reset_index(drop=True,inplace=True)


# In[92]:


df_alb


# In[93]:


date=list (range(1,3041))
df_alb.insert(1,"date",date)


# In[94]:


df_alb.dropna(inplace=True)
df_alb


# In[95]:


plt.scatter(df_alb["MinTemp"],df_alb["Rainfall"])
plt.grid()
plt.show()


# In[96]:


df_alb1=df_alb[df_alb['Rainfall']>80]
df_alb1


# In[97]:


df_alb.drop(index=[796,1156],inplace=True)


# In[98]:


plt.scatter(df_alb["MinTemp"],df_alb["Rainfall"])
plt.grid()
plt.show()


# In[99]:


x=pd.DataFrame(df_alb,columns=["MinTemp","MaxTemp","WindGustSpeed"])
y=df_alb["Rainfall"].values.reshape(-1,1)


# In[100]:


x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)


# In[101]:


regressor=LinearRegression()


# In[102]:


regressor.fit(x_train,y_train)


# In[103]:


y_pred=regressor.predict(x_test)


# In[112]:


a=x_test.MinTemp
b=y_test
c=x_test.MinTemp
d=y_pred
plt.xlabel("MinTemp")
plt.ylabel("Rainfall")
plt.scatter(a,b)
plt.plot(c,d,color="red")
plt.grid()
plt.show()


# In[105]:


x_test.insert(3,"y_test",y_test)
x_test.insert(4,"y_pred",y_pred)


# In[106]:


df2_albo=x_test.sort_values(by=['MinTemp'])
df2_albo


# In[114]:



a=df2_albo.MinTemp
b=df2_albo.y_test
c=df2_albo.MinTemp
d=df2_albo.y_pred
plt.xlabel("MinTemp")
plt.ylabel("Rainfall")
plt.scatter(a,b)
plt.plot(c,d,color="red")
plt.grid()
plt.show()


# In[115]:


print(regressor.intercept_)
print(regressor.coef_)


# In[116]:


plt.scatter(y_test,y_pred)
plt.xlabel("y_test")
plt.ylabel("y_pred")
plt.grid()
plt.show()


# In[117]:


compare=pd.DataFrame({'MinTemp':y_test.flatten(),"MinTemp pred":y_pred.flatten()})
compare


# In[118]:


print("mean Absolute Error:",metrics.mean_absolute_error(y_test,y_pred))
print("mean squared Error:",metrics.mean_squared_error(y_test,y_pred))
print("Root meansquared Error:",np.sqrt(metrics.mean_squared_error(y_test,y_pred)))
print("R2 score:",metrics.r2_score(y_test,y_pred))


# In[ ]:




