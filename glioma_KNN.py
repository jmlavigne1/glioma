"""
Spyder Editor

author: Joe LaVigne
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

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



#print(glioma_grading_clinical_and_mutation_features.data[0])

#print(y)

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier


classifier = KNeighborsClassifier(n_neighbors = 3)

#data to be cleaned up
training_data, validation_data, training_labels, validation_labels = train_test_split(X, y, test_size= 0.2, random_state=100)
