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
df_black

df_black = df_black.drop(['Race','NOTCH1', 'CSMD3', 'SMARCA4'], axis=1, inplace=False)
df_black
print(df_black.shape)

#mean_age_black = df_black.Age_at_diagnosis.mean()
#print('The mean age at diagnosis for black or african american patients is', mean_age_black)

#df_black['Age_at_diagnosis'].max()
#df_black['Age_at_diagnosis'].min()

#df_black.isna().sum()

#plt.subplots(figsize=(15,10))
#df_black['Age_at_diagnosis'].plot(kind='hist')
#plt.show()
#plt.clf()


#df_black['Grade'].value_counts().plot(kind='bar')
#plt.xlabel('Grade', weight='bold')
#plt.ylabel('Count')
#plt.show()
#plt.clf()


#pd.crosstab(df_black.ATRX, df_black.Grade).plot(kind='bar')
#plt.show()
#plt.clf()

#pd.crosstab(df_black.Gender, df_black.Grade).plot(kind='bar')
#plt.show()




#df_black = df_black.dropna()
#df_black


df_black_corr = df_black.corr()
df_black_corr

plt.subplots(figsize=(20,10))
sns.heatmap(df_black_corr, annot=True, cmap='RdBu')
plt.show()




from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

x = df_black.drop(columns=['Grade'])
y = df_black['Grade']

x.shape, y.shape

x_train, x_test, y_train,y_test = train_test_split(x,y,test_size=0.20,random_state=42)

scaler = StandardScaler()


x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)


clf = LogisticRegression()
clf.fit(x_train, y_train)

score_on_test_data = clf.score(x_test, y_test)
print(f'Test data accuracy: {score_on_test_data*100}')

#this shows us that the accuracy of the sub dataset is not very accurate.
#hyperparameter tuning

from sklearn.model_selection import RandomizedSearchCV

distributions = {'penalty': ['l1', 'l2', 'elasticnet'], 'max_iter': range(10, 50), 'warm_start': [True,False], 'solver':['lbfgs', 'liblinear', 'newton-cg', 'newton-cholesky', 'sag'], 'C': np.logspace(-1,1,22)}

clf = RandomizedSearchCV(estimator=LogisticRegression(), param_distributions = distributions, n_iter=100, scoring='accuracy', n_jobs=-1, verbose=1, random_state=42,)
clf.fit(x_train, y_train)
best_params = clf.best_params_
clf_best_dt = LogisticRegression(**best_params)
clf_best_dt.fit(x_train, y_train)
accuracy_score_2 = clf_best_dt.score(x_test, y_test)
print(accuracy_score_2*100)

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

y_pred = clf_best_dt.predict(x_test)

disp = ConfusionMatrixDisplay.from_predictions(y_test, y_pred, cmap=plt.cm.afmhot, display_labels = clf_best_dt.classes_)

fig = disp.ax_.get_figure()
fig.set_figwidth(15)
fig.set_figheight(10)
plt.title('Confusion Matrix', fontsize=20)
plt.show()

from sklearn.metrics import roc_curve, roc_auc_score
####



