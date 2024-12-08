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


plt.scatter(x=X, y=y, color='black')
plt.show()
