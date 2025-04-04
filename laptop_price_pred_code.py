# -*- coding: utf-8 -*-
"""laptop_price_pred.ipynb

Automatically generated by Colab.

Original file is located at 
    https://colab.research.google.com/drive/117mPfkkVAinQZL584KfIny3qKiMNznSo
"""

# Commented out IPython magic to ensure Python compatibility.
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
# %matplotlib inline

laptop_data = pd.read_csv('./laptop_data.csv')

laptop_data.head()

laptop_data = laptop_data.drop(columns = 'Unnamed: 0',axis = 1)

laptop_data.head()

laptop_data.duplicated().sum()

laptop_data.isnull().sum()

laptop_data.head()

laptop_data['Ram'] = laptop_data['Ram'].str.replace('GB',' ')

laptop_data['Weight'] = laptop_data['Weight'].str.replace('kg',' ')

laptop_data.head()

laptop_data['Ram'].dtype

laptop_data['Weight'].dtype

laptop_data['Ram'] = laptop_data['Ram'].astype(np.int32)

laptop_data['Weight'] = laptop_data['Weight'].astype(np.float64)

laptop_data.info()

sns.displot(laptop_data['Price'])

laptop_data['Company'].value_counts().plot(kind = 'bar')

sns.barplot(x = laptop_data['Company'],y = laptop_data['Price'])
plt.xticks(rotation = 'vertical')
plt.show()

laptop_data['TypeName'].value_counts().plot(kind = 'bar')

sns.barplot(x = laptop_data['TypeName'],y = laptop_data['Price'])
plt.xticks(rotation = 'vertical')
plt.show()

sns.displot(laptop_data['Inches'])

sns.scatterplot(x= laptop_data['Inches'],y = laptop_data['Price' ])

laptop_data['ScreenResolution'].value_counts()

laptop_data['TouchScreen'] = laptop_data['ScreenResolution'].apply(lambda x:1 if 'Touchscreen' in x else 0)

laptop_data.head()

laptop_data['TouchScreen'].value_counts().plot(kind = 'bar')

laptop_data['Ips'] = laptop_data['ScreenResolution'].apply(lambda x:1 if 'IPS' in x else 0)

laptop_data.head()

sns.barplot(x = laptop_data['Ips'],y = laptop_data['Price'])

new = laptop_data['ScreenResolution'].str.split('x',n=1,expand = True)

laptop_data['x_res'] = new[0]
laptop_data['y_res'] = new[1]

laptop_data.head()

laptop_data.sample(5)

laptop_data['x_res'] = laptop_data['x_res'].str.replace(',',' ').str.findall(r'(\d+\.?\d+)').apply(lambda x:x[0])

laptop_data.head()

laptop_data.info()

laptop_data['x_res']= laptop_data['x_res'].astype(np.int32)
laptop_data['y_res']= laptop_data['y_res'].astype(np.int32)

laptop_data.info()

numeric_data = laptop_data.select_dtypes(include=np.number)
correlation_matrix = numeric_data.corr()
price_correlation = correlation_matrix['Price']
print(price_correlation)

laptop_data['ppi'] = ((laptop_data['x_res']**2) +(laptop_data['y_res']**2))**0.5/laptop_data['Inches'].astype(float)

numeric_data = laptop_data.select_dtypes(include=np.number)
correlation_matrix = numeric_data.corr()
price_correlation = correlation_matrix['Price']
print(price_correlation)

laptop_data.drop(columns = ['ScreenResolution','Inches','x_res','y_res'],inplace = True)

laptop_data.head()

laptop_data['Cpu'].value_counts()

laptop_data['Cpu_name'] = laptop_data['Cpu'].apply(lambda x:" ".join(x.split()[0:3]))

laptop_data.head()

def fetch_processor(text):
  if text == 'Intel Core i7' or text == 'Intel Core i5' or text == 'Intel Core i3':
    return text
  else:
      if text.split()[0] == 'Intel':
        return 'Other Intel Processor'
      else:
        return 'AMD Processor'

laptop_data.head()

laptop_data['Cpu Brand'] = laptop_data['Cpu_name'].apply(fetch_processor)

laptop_data.head()

laptop_data['Cpu Brand'].value_counts().plot(kind = 'bar')

sns.barplot(x = laptop_data['Cpu Brand'],y = laptop_data['Price'])
plt.xticks(rotation = 'vertical')
plt.show()

laptop_data.drop(columns = ['Cpu','Cpu_name'],inplace = True)

laptop_data.head()

laptop_data['Ram'].value_counts().plot(kind = 'bar')

sns.barplot(x = laptop_data['Ram'],y = laptop_data['Price'])
plt.xticks(rotation = 'vertical')
plt.show()

laptop_data['Memory'].value_counts()

laptop_data['Memory'] = laptop_data['Memory'].astype(str).replace('\.0', '', regex=True)
laptop_data['Memory'] = laptop_data['Memory'].str.replace('GB', '')
laptop_data['Memory'] = laptop_data['Memory'].str.replace('TB','000')
laptop_data['Memory'] = laptop_data['Memory'].str.replace(r'\D', '', regex=True) # Removing all non-numeric characters
laptop_data['Memory'] = pd.to_numeric(laptop_data['Memory'], errors='coerce').fillna(0).astype(int) # Convert to numeric, handle errors and fill NaN with 0

# new = laptop_data['Memory'].str.split("+",n = 1,expand = True)

laptop_data['first']= new[0]
laptop_data['first']= laptop_data['first'].str.strip()

laptop_data['second']= new[1]

laptop_data['Layer1HDD'] = laptop_data['first'].apply(lambda x: 1 if 'HDD' in x else 0)
laptop_data['Layer1SSD'] = laptop_data['first'].apply(lambda x: 1 if 'SSD' in x else 0)
laptop_data['Layer1Hybrid'] = laptop_data['first'].apply(lambda x: 1 if 'Hybrid' in x else 0)
laptop_data['Layer1Flash_Storage'] = laptop_data['first'].apply(lambda x: 1 if 'Flash Storage' in x else 0)

# Removing all non-numeric characters before converting to int
laptop_data['first'] = laptop_data['first'].str.replace(r'\D', '', regex=True) # regex=True enables using r'\D'

laptop_data['second'].fillna("0",inplace = True)

laptop_data['Layer2HDD'] = laptop_data['second'].apply(lambda x: 1 if 'HDD' in x else 0)
laptop_data['Layer2SSD'] = laptop_data['second'].apply(lambda x: 1 if 'SSD' in x else 0)
laptop_data['Layer2Hybrid'] = laptop_data['second'].apply(lambda x: 1 if 'Hybrid' in x else 0)
laptop_data['Layer2Flash_Storage'] = laptop_data['second'].apply(lambda x: 1 if 'Flash Storage' in x else 0)

laptop_data['second']= laptop_data['second'].str.replace(r'\D', '', regex=True) # regex=True enables using r'\D'

laptop_data['first'] = laptop_data['first'].astype(int)
laptop_data['second'] = laptop_data['second'].astype(int)

laptop_data['HDD']=(laptop_data['first']*laptop_data['Layer1HDD']+laptop_data['second']*laptop_data['Layer2HDD'])
laptop_data['SSD']=(laptop_data['first']*laptop_data['Layer1SSD']+laptop_data['second']*laptop_data['Layer2SSD'])
laptop_data['Hybrid']=(laptop_data['first']*laptop_data['Layer1Hybrid']+laptop_data['second']*laptop_data['Layer2Hybrid'])
laptop_data['FLash_Storage']=(laptop_data['first']*laptop_data['Layer1Flash_Storage']+laptop_data['second']*laptop_data['Layer2Flash_Storage'])

laptop_data.drop(columns = ['first','second','Layer1HDD','Layer1SSD','Layer1Hybrid','Layer1Flash_Storage','Layer2HDD','Layer2SSD','Layer2Hybrid','Layer2Flash_Storage'],inplace = True)

laptop_data.head()

laptop_data.drop(columns = ['Memory'],inplace = True)

# numeric_data = laptop_data.select_dtypes(include=np.number)
# correlation_matrix = numeric_data.corr()
# price_correlation = correlation_matrix['Price']
# print(price_correlation)

numeric_data = laptop_data.select_dtypes(np.number)
correlation_matrix = numeric_data.corr()
price_correlation = correlation_matrix['Price']
print(price_correlation)

laptop_data.info()

laptop_data.drop(columns = ['Hybrid','FLash_Storage'],inplace = True)

laptop_data.head()

laptop_data['Gpu'].value_counts(   )

laptop_data['Gpu Brand'] = laptop_data['Gpu'].apply(lambda x:x.split()[0])

laptop_data.head()

laptop_data['Gpu Brand'].value_counts()

laptop_data = laptop_data[laptop_data['Gpu Brand'] != 'ARM']

laptop_data['Gpu Brand'].value_counts()

sns.barplot(x = laptop_data['Gpu Brand'],y  = laptop_data['Price'],estimator = np.median)
plt.xticks(rotation = 'vertical')
plt.show()

laptop_data.drop(columns = ['Gpu'],inplace = True)

laptop_data.head()

laptop_data['OpSys'].value_counts()

sns.barplot(x = laptop_data['OpSys'],y = laptop_data['Price'])
plt.xticks(rotation = 'vertical')
plt.show()

def cat_os(inp):
  if inp == 'Windows 10' or inp == 'Windows 7' or inp == 'windows 10 S':
    return 'Windows'
  elif inp == 'macOS' or inp == 'mac OS X':
    return 'Mac'
  else:
    return 'Others/No OS/Linux'

laptop_data['os'] = laptop_data['OpSys'].apply(cat_os)

laptop_data.head()

laptop_data.drop(columns = ['OpSys'],inplace = True)

sns.barplot(x = laptop_data['os'],y = laptop_data['Price'])
plt.xticks(rotation = 'vertical')
plt.show()

sns.displot(laptop_data['Weight'])

sns.scatterplot(x = laptop_data['Weight'],y = laptop_data['Price'])

numeric_data = laptop_data.select_dtypes(include=np.number)
correlation_matrix = numeric_data.corr()
price_correlation = correlation_matrix['Price']
print(price_correlation)

sns.heatmap(correlation_matrix)

sns.displot(laptop_data['Price'])

sns.displot(np.log(laptop_data['Price']))

x = laptop_data.drop(columns = 'Price')
y = np.log(laptop_data['Price'])

x

y

from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test = train_test_split(x,y ,test_size = 0.2,random_state=42)

from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score,mean_absolute_error

from sklearn.linear_model import LinearRegression,Ridge,Lasso
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor,GradientBoostingRegressor,ExtraTreesRegressor,AdaBoostRegressor
from sklearn.svm import SVR
from xgboost import XGBRegressor

step1 = ColumnTransformer(transformers=[
    ('col_tnf',OneHotEncoder(sparse_output = False,handle_unknown='ignore',drop = 'first' ),[0,1,7,10,11])
],remainder = 'passthrough')

step2 = LinearRegression()

pipe = Pipeline([
    ('step1',step1),
    ('step2',step2)
])

pipe.fit(x_train,y_train)

y_pred = pipe.predict(x_test)

print('R2 score',r2_score(y_test,y_pred))
print('mean absolute error',mean_absolute_error(y_test,y_pred))

laptop_data.head()

step1 = ColumnTransformer(transformers=[
    ('col_tnf',OneHotEncoder(sparse_output = False,handle_unknown='ignore',drop = 'first' ),[0,1,7,10,11])
],remainder = 'passthrough')

step2 = Ridge(alpha =10)

pipe = Pipeline([
    ('step1',step1),
    ('step2',step2)
])

pipe.fit(x_train,y_train)

y_pred = pipe.predict(x_test)

print('R2 score',r2_score(y_test,y_pred))
print('mean absolute error',mean_absolute_error(y_test,y_pred))

step1 = ColumnTransformer(transformers=[
    ('col_tnf',OneHotEncoder(sparse_output = False,handle_unknown='ignore',drop = 'first' ),[0,1,7,10,11])
],remainder = 'passthrough')

step2 = Lasso(alpha = 0.001)

pipe = Pipeline([
    ('step1',step1),
    ('step2',step2)
])

pipe.fit(x_train,y_train)

y_pred = pipe.predict(x_test)

print('R2 score',r2_score(y_test,y_pred))
print('mean absolute error',mean_absolute_error(y_test,y_pred))

step1 = ColumnTransformer(transformers=[
    ('col_tnf',OneHotEncoder(sparse_output = False,handle_unknown='ignore',drop = 'first' ),[0,1,7,10,11])
],remainder = 'passthrough')

step2 = KNeighborsRegressor(n_neighbors= 3)

pipe = Pipeline([
    ('step1',step1),
    ('step2',step2)
])

pipe.fit(x_train,y_train)

y_pred = pipe.predict(x_test)

print('R2 score',r2_score(y_test,y_pred))
print('mean absolute error',mean_absolute_error(y_test,y_pred))

step1 = ColumnTransformer(transformers=[
    ('col_tnf',OneHotEncoder(sparse_output = False,handle_unknown='ignore',drop = 'first' ),[0,1,7,10,11])
],remainder = 'passthrough')

step2 = DecisionTreeRegressor(max_depth= 10)

pipe = Pipeline([
    ('step1',step1),
    ('step2',step2)
])

pipe.fit(x_train,y_train)

y_pred = pipe.predict(x_test)

print('R2 score',r2_score(y_test,y_pred))
print('mean absolute error',mean_absolute_error(y_test,y_pred))

step1 = ColumnTransformer(transformers=[
    ('col_tnf',OneHotEncoder(sparse_output = False,handle_unknown='ignore',drop = 'first' ),[0,1,7,10,11])
],remainder = 'passthrough')

step2 = SVR(kernel = 'rbf',C = 10000,epsilon=0.1)

pipe = Pipeline([
    ('step1',step1),
    ('step2',step2)
])

pipe.fit(x_train,y_train)

y_pred = pipe.predict(x_test)

print('R2 score',r2_score(y_test,y_pred))
print('mean absolute error',mean_absolute_error(y_test,y_pred))

step1 = ColumnTransformer(transformers=[
    ('col_tnf',OneHotEncoder(sparse_output = False,handle_unknown='ignore',drop = 'first' ),[0,1,7,10,11])
],remainder = 'passthrough')

step2 =RandomForestRegressor(n_estimators= 100,
                             random_state=3,
                             max_samples = 0.5,
                             max_features=0.75,
                             max_depth = 15)

pipe = Pipeline([
    ('step1',step1),
    ('step2',step2)
])

pipe.fit(x_train,y_train)

y_pred = pipe.predict(x_test)

print('R2 score',r2_score(y_test,y_pred))
print('mean absolute error',mean_absolute_error(y_test,y_pred))

step1 = ColumnTransformer(transformers=[
    ('col_tnf',OneHotEncoder(sparse_output = False,handle_unknown='ignore',drop = 'first' ),[0,1,7,10,11])
],remainder = 'passthrough')

step2 =ExtraTreesRegressor(n_estimators= 100,
                             random_state=3,
                             max_samples = None,
                             max_features=0.75,
                             max_depth = 15)

pipe = Pipeline([
    ('step1',step1),
    ('step2',step2)
])

pipe.fit(x_train,y_train)

y_pred = pipe.predict(x_test)

print('R2 score',r2_score(y_test,y_pred))
print('mean absolute error',mean_absolute_error(y_test,y_pred))

step1 = ColumnTransformer(transformers=[
    ('col_tnf',OneHotEncoder(sparse_output = False,handle_unknown='ignore',drop = 'first' ),[0,1,7,10,11])
],remainder = 'passthrough')

step2 = GradientBoostingRegressor(n_estimators= 100,
                             random_state=3,
                             max_features=0.75,
                             max_depth = 15)
pipe = Pipeline([
    ('step1',step1),
    ('step2',step2)
])

pipe.fit(x_train,y_train)

y_pred = pipe.predict(x_test)

print('R2 score',r2_score(y_test,y_pred))
print('mean absolute error',mean_absolute_error(y_test,y_pred))

step1 = ColumnTransformer(transformers=[
    ('col_tnf',OneHotEncoder(sparse_output = False,handle_unknown='ignore',drop = 'first' ),[0,1,7,10,11])
],remainder = 'passthrough')

step2 =AdaBoostRegressor(n_estimators =15,learning_rate = 1.0)

pipe = Pipeline([
    ('step1',step1),
    ('step2',step2)
])

pipe.fit(x_train,y_train)

y_pred = pipe.predict(x_test)

print('R2 score',r2_score(y_test,y_pred))
print('mean absolute error',mean_absolute_error(y_test,y_pred))

step1 = ColumnTransformer(transformers=[
    ('col_tnf',OneHotEncoder(sparse_output = False,handle_unknown='ignore',drop = 'first' ),[0,1,7,10,11])
],remainder = 'passthrough')

step2 =XGBRegressor(n_estimators = 45,max_depth = 5,learning_rate = 0.5)
pipe = Pipeline([
    ('step1',step1),
    ('step2',step2)
])

pipe.fit(x_train,y_train)

y_pred = pipe.predict(x_test)

print('R2 score',r2_score(y_test,y_pred))
print('mean absolute error',mean_absolute_error(y_test,y_pred))

from sklearn.ensemble import VotingRegressor,StackingRegressor

step1 = ColumnTransformer(transformers=[
    ('col_tnf',OneHotEncoder(sparse_output = False,handle_unknown='ignore',drop = 'first' ),[0,1,7,10,11])
],remainder = 'passthrough')

rf = RandomForestRegressor(n_estimators= 100,random_state=3,max_samples = 0.5,max_features=0.75,max_depth = 15)
gbdt = GradientBoostingRegressor(n_estimators= 100,random_state=3,max_features=0.75,max_depth = 15)
xgb = XGBRegressor(n_estimators = 45,max_depth = 5,learning_rate = 0.5)
et = ExtraTreesRegressor(n_estimators= 100,random_state=3,max_samples = None,max_features=0.75,max_depth = 15)

step2 =VotingRegressor([('rf',rf),('gbdt',gbdt),('xgb',xgb),('et',et)],weights = [5,1,1,1])

pipe = Pipeline([
    ('step1',step1),
    ('step2',step2)
])

pipe.fit(x_train,y_train)

y_pred = pipe.predict(x_test)

print('R2 score',r2_score(y_test,y_pred))
print('mean absolute error',mean_absolute_error(y_test,y_pred))

from sklearn.ensemble import VotingRegressor,StackingRegressor

step1 = ColumnTransformer(transformers=[
    ('col_tnf',OneHotEncoder(sparse_output = False,handle_unknown='ignore',drop = 'first' ),[0,1,7,10,11])
],remainder = 'passthrough')

estimators = [
    ('rf', RandomForestRegressor(n_estimators= 100,random_state=3,max_samples = 0.5,max_features=0.75,max_depth = 15)),
    ('gbdt',GradientBoostingRegressor(n_estimators= 100,random_state=3,max_features=0.75,max_depth = 15)),
    ('xgb', XGBRegressor(n_estimators = 45,max_depth = 5,learning_rate = 0.5))
]

step2 =VotingRegressor([('rf',rf),('gbdt',gbdt),('xgb',xgb),('et',et)],weights = [5,1,1,1])

pipe = Pipeline([
    ('step1',step1),
    ('step2',step2)
])

pipe.fit(x_train,y_train)

y_pred = pipe.predict(x_test)

print('R2 score',r2_score(y_test,y_pred))
print('mean absolute error',mean_absolute_error(y_test,y_pred))

import pickle
pickle.dump(laptop_data,open('laptop_data.pkl','wb'))
pickle.dump(pipe,open('pipe.pkl','wb'))
