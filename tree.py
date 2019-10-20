import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from utils import get_data
import os
# %% imports data and separates training data
train, test = get_data()
x = train.iloc[:,1:]
y = train.iloc[:,0]
# %% create fitting model and run prediction on test data
clf = DecisionTreeClassifier()
clf.fit(x,y)
prediction = clf.predict(test)
# %% select random numbers from 'test' to plot image and show prediction
ranum = np.random.randint(0,28000,15)
for i in ranum:
    d=test.iloc[i].values
    d.shape=(28,28)
    print('Prediction: ', prediction[i])
    plt.imshow(255-d,cmap='gray')
    plt.show()
# %% creates output data and write csv
label = range(1, len(prediction) + 1)
df = pd.DataFrame(data = zip(label, prediction),columns = ['ImageId','Label'])
os.chdir(r'../submissions')
df.to_csv('tree-py.csv',index=False)