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



# remove Nan values from dataset, and sift through whether using age, gender or other marked values for our sigmoid function. 
X_race= pd.get_dummies(X['Race'])
X_race = X_race.astype(int)
print(X_race)

X = X.drop('Race', axis=1, inplace=False)
print(X)


X = X.join(X_race)
print(X)
#target(y) should be the y axis and dictate the 0 or 1 classification. For feature 'Race' I can use get_dummies to make change categorical into numerical output. 


X = X['Age_at_diagnosis']
y = y['Grade']


#################
from sklearn.linear_model import LogisticRegression
model = LogisticRegression()
model.fit(X.values,y.values)

# Plug sample data into fitted model
sample_x = np.linspace(-16.65, 33.35, 300).reshape(-1,1)
probability = model.predict_proba(sample_x[:,0])

# Plot logistic curve
plt.plot(sample_x, probability, color='red', linewidth=3)
plt.show()
##########

plt.scatter(x=X['Age_at_diagnosis'], y=y, color='black', s=100)
plt.plot(sample_x, probability, color='red', linewidth=3)
plt.ylabel('Probability glioma diagnosis')
plt.xlabel('Age at Diagnosis')
plt.title('Probability of Glioma Diagnosis by Age', weight='bold')
plt.show()
