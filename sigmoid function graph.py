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

X.shape, y.shape

# remove Nan values from dataset, and sift through whether using age, gender or other marked values for our sigmoid function. 


print(X.columns)
#target(y) should be the y axis and dictate the 0 or 1 classification. For feature 'Race' I can use get_dummies to make change categorical into numerical output. 





plt.scatter(x=X['Age_at_diagnosis'], y=y, color='black', s=100)
plt.ylabel('Probability glioma diagnosis')
plt.xlabel('Age at Diagnosis')
plt.title('Probability of Glioma Diagnosis by Age', weight='bold')
plt.show()
