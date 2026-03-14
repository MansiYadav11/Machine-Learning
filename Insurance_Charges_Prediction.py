import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings('ignore')

df=pd.read_csv('/insurance.csv')
data.head()

#EDA

df.shape
df.info()
df.describe()
df.isnull().sum()

#visulaisation

df.columns
numeric_columns=['age', 'bmi', 'children', 'charges']
for col in numeric_columns:
  plt.figure(figsize=(6,4))
  sns.histplot(df[col],kde=True,bins=20)

#visulaization individually

sns.countplot(x=df['children'])
sns.countplot(x=df['sex'])
sns.countplot(x=df['smoker'])

#inpput,output analysis

#identifying outliers
for col in numeric_columns:
  plt.figure(figsize=(6,4))
  sns.boxplot(x=df[col])

#for visualizing correlation
plt.figure(figsize=(8,6))
sns.heatmap(df.corr(numeric_only=True), annot=True) #it will only be calculated for numeric values

#  DATA CLEANING AND PREPROCESSING
df_clean=df.copy()
df_clean.head()

#1)There are no null values, so removing duplicates
df_clean.shape
df_clean.drop_duplicates(inplace=True)
df_clean.shape
df_clean.dtypes
df_clean['sex'].value_counts() #seeing if there exist more than 2 sex

#2)coverting datatype - encoding
df_clean['sex']=df_clean['sex'].map({"male":0,"female":1})
df_clean.head()

df_clean['smoker'].value_counts()
df_clean['smoker']=df_clean['smoker'].map({"no":0,"yes":1})
df_clean.head()

