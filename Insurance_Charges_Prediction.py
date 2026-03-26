import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings('ignore')

df=pd.read_csv('/insurance.csv')
df.head()

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

df_clean.rename(columns={
  'sex' : 'is_female',
  'smoker' : 'is_smoker'
  } ,inplace=True)

df_clean.head()

# Add the 'region' column back to df_clean from the original df
df_clean['region'] = df['region']

#encoding region with onehot encoding
df['region'].value_counts()

df_clean=pd.get_dummies(df_clean,columns=['region'],drop_first=True)
df_clean=df_clean.astype(int)
df_clean.head()

#FEATURE ENGINEERING AND EXTRACTION
sns.histplot(df['bmi'])
df_clean['bmi_category']=pd.cut(
    df_clean['bmi'],
    bins=[0,18.5,24.9,29.9,float('inf')]
    ,labels=['underweight','healthy','overweight','obese']
)

df_clean=pd.get_dummies(df_clean,columns=['bmi_category'],drop_first=True)

df_clean=df_clean.astype(int)
df_clean

#FEATURE SCALING
#like age and charges,children
from sklearn.preprocessing import StandardScaler
cols=['age','bmi','children']
scaler=StandardScaler()
df_clean[cols]=scaler.fit_transform(df_clean[cols])
df_clean.head()

#output variable=charges needs to be kept same

#FEATURE EXTRACTION
#by pearson Correlation Calculation

from scipy.stats import pearsonr
#Pearson Correlation Calculation#List of features t ocheck against target

selected_features=['age','bmi','children','is_female','is_smoker','region_northwest','region_southeast',
                   'region_southwest','bmi_category_Normal','bmi_category_overweight','bmi_category_obese']

correlations={
    feature:pearsonr(df_clean[feature],df_clean['charges'])[0] 
    for feature in selected_features
}

#Converting the dictionary obtained from the above code into a dataframe
correlation_df=pd.DataFrame(list(correlations.items()),columns=['feature','correlation'])
correlation_df=correlation_df.sort_values(by='correlation',ascending=False)
correlation_df.head()

#Selecting categorical features for chi test
cat_features={
    'is_female','is_smoker',
    'region_northwest','region_southeast','region_southwest,
    'bmi_category_healthy','bmi_category_overweight','bmi_category_obese'
}
#The target variable is a continuous variable(charges) not categorical
#So converting the charges into bins (to reduce the continuity and bringing the categorical nature )

from scipy.stats import chi2_contingency
import pandas as pd

alpha = 0.05
df_clean['charges_bin']=pd.cut(df_clean['charges'],bins=4,labels=False)
chi2_results = {}

for col in cat_features:
    contingency_table = pd.crosstab(df_clean[col], df_clean['charges_bin'])
    chi2_stat, p_val, _, _ = chi2_contingency(contingency)
    decision = 'Reject Null (Keep Feature)' if p_val <alpha else 'Accept Null Feature)'
    chi2_results[col]={
        'chi2_statistis':chi2_stat,
        'p_value':p_val,
        'Decision':decision
    }
    
chi2_df=pd.DataFrame(chi2_results).T
chi2_df=chi2_df.sort_values(by='p_value')
chi2_df

final_df=df_clean[['age','is_female','bmi','children','is_smoker','charges','region_southeast','bmi_category_Obese']]
final_df.head()
