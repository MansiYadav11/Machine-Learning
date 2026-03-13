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


