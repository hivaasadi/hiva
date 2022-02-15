#!/usr/bin/env python
# coding: utf-8

# In[937]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics


# In[938]:


data=pd.read_csv("cardata.csv")
df=pd.DataFrame (data.drop('Car_Name',axis=1))
df


# In[939]:


df.isnull().sum()


# In[940]:


# int به strتبديل کردن ستون هايي با مقادير   
df['Fuel_Type'].replace({'Petrol':0, 'Diesel':1,'CNG':2 }, inplace=True)
df['Seller_Type'].replace({'Dealer':0, 'Individual':1 }, inplace=True)
df['Transmission'].replace({'Manual':0, 'Automatic':1 }, inplace=True)


# In[941]:


#ساخت يک ستون براي سن ماشين ها بر اساس سال ساخت 2018
for i in df.index:
    j=2018-df['Year']
    
print(j)


# In[942]:


#اضافه کردن ستون (سال) به ديتاست
df.insert(1,"Age",j,True)
df


# In[943]:


df.describe()


# In[944]:


#رسم کاونتپلات براي تمام ستون ها
plt.figure(figsize=(8,5),dpi=90)
sns.countplot(x="Fuel_Type",data=df)
plt.xticks(rotation=0,fontsize=15)
plt.yticks(fontsize=10)
plt.xlabel("Fuel_Type",fontsize=15)
plt.ylabel("count",fontsize=15)
plt.title("count of Fuel_Type",fontsize=15)
plt.grid()


# In[945]:


plt.figure(figsize=(8,5),dpi=90)
sns.countplot(x="Year",data=df)
plt.xticks(rotation=90,fontsize=15)
plt.yticks(fontsize=10)
plt.xlabel("Year",fontsize=25)
plt.ylabel("count",fontsize=25)
plt.title("count of Year",fontsize=25)
plt.grid()


# In[946]:


plt.figure(figsize=(12,5),dpi=90)
sns.countplot(x="Age",data=df)
plt.xticks(rotation=0,fontsize=15)
plt.yticks(fontsize=10)
plt.xlabel("Age",fontsize=25)
plt.ylabel("count",fontsize=25)
plt.title("count of Age",fontsize=25)
plt.grid()


# In[947]:


plt.figure(figsize=(35,15),dpi=90)
sns.countplot(x="Selling_Price",data=df)
plt.xticks(rotation=90,fontsize=15)
plt.yticks(fontsize=35)
plt.xlabel("Selling_Price",fontsize=25)
plt.ylabel("count",fontsize=35)
plt.title("count of Selling_Price",fontsize=35)
plt.grid()


# In[948]:


plt.figure(figsize=(35,15),dpi=90)
sns.countplot(x="Present_Price",data=df)
plt.xticks(rotation=90,fontsize=15)
plt.yticks(fontsize=35)
plt.xlabel("Present_Price",fontsize=25)
plt.ylabel("count",fontsize=35)
plt.title("count of Present_Price",fontsize=35)
plt.grid()


# In[949]:


plt.figure(figsize=(35,15),dpi=90)
sns.countplot(x="Kms_Driven",data=df)
plt.xticks(rotation=90,fontsize=15)
plt.yticks(fontsize=35)
plt.xlabel("Kms_Driven",fontsize=25)
plt.ylabel("count",fontsize=35)
plt.title("count of Kms_Driven",fontsize=35)
plt.grid()


# In[950]:


plt.figure(figsize=(8,4),dpi=90)
sns.countplot(x="Seller_Type",data=df)
plt.xticks(rotation=0,fontsize=15)
plt.yticks(fontsize=15)
plt.xlabel("Seller_Type",fontsize=15)
plt.ylabel("count",fontsize=15)
plt.title("count of Seller_Type",fontsize=15)
plt.grid()


# In[951]:


# ...براي مدلسازي رگرشن و ساخت تست و ترين و فراخواني نوع الگوريتم و x,y ساخت


# In[952]:


x=pd.DataFrame(df,columns=['Age','Present_Price','Kms_Driven','Fuel_Type','Seller_Type','Transmission','Owner'])
y=df['Selling_Price'].values.reshape(-1,1)


# In[953]:


x_train, x_test, y_train, y_test=train_test_split(x,y,test_size=0.20,random_state=0)


# In[954]:


regressor=LinearRegression()
regressor.fit(x_train,y_train)


# In[955]:


y_pred=regressor.predict(x_test)


# In[956]:


print(regressor.intercept_)
print(regressor.coef_)


# In[957]:


df


# In[958]:


# بررسي و نمايش کي فولد و بررسي کروليشن


# In[959]:


from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score


# In[962]:


model_new=LinearRegression()
kfold_val=KFold(10)
results=cross_val_score(model_new,x,y,cv=kfold_val)


# In[963]:


print(results)
print(np.mean(results))


# In[ ]:





# In[965]:


cor_Age=np.corrcoef(df.Age,df.Selling_Price)
cor_Age


# In[966]:


cor_Present_Price=np.corrcoef(df.Present_Price,df.Selling_Price)
cor_Present_Price


# In[967]:


cor_Kms_Driven=np.corrcoef(df.Kms_Driven,df.Selling_Price)
cor_Kms_Driven


# In[969]:


cor_Fuel_Type=np.corrcoef(df.Fuel_Type,df.Selling_Price)
cor_Fuel_Type


# In[970]:


cor_Seller_Type=np.corrcoef(df.Seller_Type,df.Selling_Price)
cor_Seller_Type


# In[971]:


cor_Transmission=np.corrcoef(df.Transmission,df.Selling_Price)
cor_Transmission


# In[972]:


cor_Owner=np.corrcoef(df.Owner,df.Selling_Price)
cor_Owner


# In[973]:


#نمودار اسکترپلات براي مشاهده ديتاي واقعي و ديتاي پرديکت
a=x_test.Age
b=y_test
c=x_test.Age
d=y_pred
plt.scatter(a,b)
plt.scatter(c,d)
plt.show()


# In[974]:


a=x_test.Age
b=y_test
c=x_test.Age
d=y_pred
plt.plot(a,b)
plt.plot(c,d)
plt.show()


# In[975]:


#  براي رسم پلات به صورت منظم Age مرتب کردن ستون 
x_test.insert(1,"y_test",y_test)
x_test.insert(2,"y_pred",y_pred)
dfsort_Age=x_test.sort_values(by=['Age'])
dfsort_Age


# In[976]:


plt.figure(figsize=(8,6),dpi=90)
a=dfsort_Age.Age
b=dfsort_Age.y_test
c=dfsort_Age.Age
d=dfsort_Age.y_pred
plt.xlabel("Age")
plt.ylabel("Selling_Price")
plt.scatter(a,b)
plt.plot(c,d,color="red")
plt.show()


# In[977]:


dfsort_Present_Price=x_test.sort_values(by=['Present_Price'])
dfsort_Present_Price


# In[978]:


plt.figure(figsize=(8,6),dpi=90)
a=dfsort_Present_Price.Present_Price
b=dfsort_Present_Price.y_test
c=dfsort_Present_Price.Present_Price
d=dfsort_Present_Price.y_pred
plt.xlabel("Present_Price",fontsize=25)
plt.ylabel("Selling_Price",fontsize=25)
plt.scatter(a,b)
plt.plot(c,d,color="red")
plt.show()


# In[979]:


dfsort_Kms_Driven=x_test.sort_values(by=['Kms_Driven'])
dfsort_Kms_Driven


# In[980]:


plt.figure(figsize=(8,6),dpi=90)
a=dfsort_Kms_Driven.Kms_Driven
b=dfsort_Kms_Driven.y_test
c=dfsort_Kms_Driven.Kms_Driven
d=dfsort_Kms_Driven.y_pred
plt.xlabel("Kms_Driven",fontsize=25)
plt.ylabel("Selling_Price",fontsize=25)
plt.scatter(a,b)
plt.plot(c,d,color="red")
plt.show()


# In[981]:


dfsort_Fuel_Type=x_test.sort_values(by=['Fuel_Type'])
dfsort_Fuel_Type


# In[982]:


plt.figure(figsize=(8,6),dpi=90)
a=dfsort_Fuel_Type.Fuel_Type
b=dfsort_Kms_Driven.y_test
c=dfsort_Fuel_Type.Fuel_Type
d=dfsort_Kms_Driven.y_pred
plt.xlabel("Fuel_Type",fontsize=25)
plt.ylabel("Selling_Price",fontsize=25)
plt.scatter(a,b)
plt.plot(c,d,color="red")
plt.show()


# In[983]:


dfsort_Seller_Type=x_test.sort_values(by=['Seller_Type'])
dfsort_Seller_Type


# In[984]:


plt.figure(figsize=(8,6),dpi=90)
a=dfsort_Seller_Type.Seller_Type
b=dfsort_Seller_Type.y_test
c=dfsort_Seller_Type.Seller_Type
d=dfsort_Seller_Type.y_pred
plt.xlabel("Seller_Type",fontsize=25)
plt.ylabel("Selling_Price",fontsize=25)
plt.scatter(a,b)
plt.plot(c,d,color="red")
plt.show()


# In[985]:


dfsort_Transmission=x_test.sort_values(by=['Transmission'])
dfsort_Transmission


# In[986]:


plt.figure(figsize=(8,6),dpi=90)
a=dfsort_Transmission.Transmission
b=dfsort_Transmission.y_test
c=dfsort_Transmission.Transmission
d=dfsort_Transmission.y_pred
plt.xlabel("Transmission",fontsize=25)
plt.ylabel("Selling_Price",fontsize=25)
plt.scatter(a,b)
plt.plot(c,d,color="red")
plt.show()


# In[987]:


dfsort_Owner=x_test.sort_values(by=['Owner'])
dfsort_Owner


# In[988]:


plt.figure(figsize=(8,6),dpi=90)
a=dfsort_Owner.Owner
b=dfsort_Owner.y_test
c=dfsort_Owner.Owner
d=dfsort_Owner.y_pred
plt.xlabel("Owner",fontsize=25)
plt.ylabel("Selling_Price",fontsize=25)
plt.scatter(a,b)
plt.plot(c,d,color="red")
plt.show()


# In[989]:


# y , y^ رسم نمودار اسکتر 
plt.figure(figsize=(8,6),dpi=90)
plt.scatter(y_test,y_pred)
plt.grid()
plt.xlabel("y_test")
plt.ylabel("y_pred")
plt.show()


# In[991]:


compare=pd.DataFrame({"y_test":y_test.flatten(),"y_pred":y_pred.flatten()})
compare


# In[995]:


print("mean Absolute Error:",metrics.mean_absolute_error(y_test,y_pred))
print("mean squared Error:",metrics.mean_squared_error(y_test,y_pred))
print("Root meansquared Error:",np.sqrt(metrics.mean_squared_error(y_test,y_pred)))
print("R2 score:",metrics.r2_score(y_test,y_pred))


# In[ ]:





# In[ ]:





# In[996]:


#در کيفولد تشخيص داديم که بايد در يک فولد ديتاها حذف بشن تا مدل را بيشتر بررسي کنيم 
df_1=df.iloc[:170]
df_2=df.iloc[190:]
df_3=df_1.append(df_2)
df_3.sort_index(axis=0)
df_3


# In[997]:


#با حذف اين ايندکس ها مدل بهتر شد اين ايندکس ها حالت نويز داشتن
df_3.drop(index=[196,86,64],inplace=True)
df_3


# In[998]:


www=df_3["Kms_Driven"].nlargest(5)
www


# In[999]:


x=pd.DataFrame(df_3,columns=['Age','Present_Price','Kms_Driven','Fuel_Type','Seller_Type','Transmission','Owner'])
y=df_3['Selling_Price'].values.reshape(-1,1)


# In[1000]:


x_train, x_test, y_train, y_test=train_test_split(x,y,test_size=0.14,random_state=0)


# In[1001]:


regressor=LinearRegression()
regressor.fit(x_train,y_train)


# In[1002]:


y_pred=regressor.predict(x_test)


# In[1003]:


print(regressor.intercept_)
print(regressor.coef_)


# In[1004]:


#نمودار اسکترپلات براي مشاهده ديتاي واقعي و ديتاي پرديکت
a=x_test.Age
b=y_test
c=x_test.Age
d=y_pred
plt.scatter(a,b)
plt.scatter(c,d)
plt.show()


# In[1005]:


# y , y^ رسم نمودار اسکتر 
plt.figure(figsize=(8,6),dpi=90)
plt.scatter(y_test,y_pred)
plt.grid()
plt.xlabel("y_test")
plt.ylabel("y_pred")
plt.show()


# In[1006]:


# y , y^ رسم نمودار اسکتر 
plt.figure(figsize=(8,6),dpi=90)
plt.scatter(df_3["Fuel_Type"],df_3["Selling_Price"])
plt.grid()
plt.xlabel("y_test")
plt.ylabel("y_pred")
plt.show()


# In[1007]:


compare1=pd.DataFrame({"y_test":y_test.flatten(),"y_pred":y_pred.flatten()})
compare1


# In[936]:


print("mean Absolute Error:",metrics.mean_absolute_error(y_test,y_pred))
print("mean squared Error:",metrics.mean_squared_error(y_test,y_pred))
print("Root meansquared Error:",np.sqrt(metrics.mean_squared_error(y_test,y_pred)))
print("R2 score:",metrics.r2_score(y_test,y_pred))


# In[1008]:


df_3


# In[1015]:


print(regressor.intercept_)
print(regressor.coef_)


# In[994]:


#y=3.15+(-4.02)+4.46+(-6.09)+1.42+(-1.08)+1.37+(-6.83)
#df2=pd.DataFrame({"Age":[],"Present_Price":[],"Kms_Driven":[],
                  #"Fuel_Type":[],"Seller_Type":[],"Transmission":[],"Owner":[]})


# In[1017]:


Age=float(input("input Age="))
Present_Price=float(input("input Present_Price="))
Kms_Driven=float(input("input Kms_Driven"))
Fuel_Type=float(input("input Fuel_Type"))
Seller_Type=float(input("input Seller_Type"))
Transmission=float(input("input Transmission"))
Owner=float(input("input Owner"))
print(2.79+(-2.76*Age)+(5.33*Present_Price)+(-2.94*Kms_Driven)+
      (1.35*Fuel_Type)+(-8.07*Seller_Type)+(5.03*Transmission)+(-7.48*Owner))


# In[ ]:




