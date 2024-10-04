# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

panc_py = pd.read_csv(r"C:\Users\lavig\Desktop\cancer_panc\cancer_panc\Debernardi et al 2020 data.csv")



plt.scatter(x= "REG1B", y="sex", data= panc_py, color="black")
plt.ylabel('sex')
plt.xlabel('REG1A ng/ml level')
plt.show()



REG1B = panc_py[["REG1B"]]
diagnosis = panc_py[["diagnosis"]]
diagnosis= np.array(diagnosis)
diagnosis = diagnosis.reshape(-1)

from sklearn.linear_model import LogisticRegression
model=LogisticRegression()
model.fit(REG1B, diagnosis)

sample_x = np.linspace(-16.65, 33.35, 300).reshape(-1,1)
probability = model.predict_proba(sample_x)[:,1]


plt.plot(sample_x, probability, color='red', linewidth=3)
