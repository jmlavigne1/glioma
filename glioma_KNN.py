"""
Spyder Editor

author: Joe LaVigne
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from ucimlrepo import fetch_ucirepo 
  
# fetch dataset 
glioma_grading_clinical_and_mutation_features = fetch_ucirepo(id=759) 
  
# data (as pandas dataframes) 
X = glioma_grading_clinical_and_mutation_features.data.features 
y = glioma_grading_clinical_and_mutation_features.data.targets 
  
# metadata 
#print(glioma_grading_clinical_and_mutation_features.metadata) 
  
# variable information 
#print(glioma_grading_clinical_and_mutation_features.variables) 

print(X)

print(y)

df = X.join(y)
df


#EDA
df_black = df[df['Race'] == 'black or african american']
df_black.shape

mean_age_black = df_black.Age_at_diagnosis.mean()
print('The mean age at diagnosis for black or african american patients is', mean_age_black)

df_black['Age_at_diagnosis'].max()
df_black['Age_at_diagnosis'].min()

df_black.isna().sum()

plt.subplots(figsize=(15,10))
df_black['Age_at_diagnosis'].plot(kind='hist')
plt.show()
plt.clf()


df_black['Grade'].value_counts().plot(kind='bar')
plt.xlabel('Grade', weight='bold')
plt.ylabel('Count')
plt.show()
plt.clf()


pd.crosstab(df_black.ATRX, df_black.Grade).plot(kind='bar')
plt.show()
plt.clf()

pd.crosstab(df_black.Gender, df_black.Grade).plot(kind='bar')
plt.show()

df_black = df_black.drop(columns=[['Race', 'NOTCH1', 'CSMD3', 'SMARCA4']])
df_black.shape


df_black = df_black.dropna()
df_black.shape


df_black_corr = df_black.corr()
df_black_corr

plt.subplots(figsize=(20,10))
sns.heatmap(df_black_corr, annot=True, cmap='RdBu')
plt.show()




from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression




#data to be cleaned up
#training_data, validation_data, training_labels, validation_labels = train_test_split(X, y, test_size= 0.2, random_state=100)
