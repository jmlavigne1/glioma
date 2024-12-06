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

df_black.isna().sum()





from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression




#data to be cleaned up
#training_data, validation_data, training_labels, validation_labels = train_test_split(X, y, test_size= 0.2, random_state=100)
