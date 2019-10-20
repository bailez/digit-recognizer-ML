import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from utils import get_data

import os
train, test = get_data()
# %%
clf = DecisionTreeClassifier()
# %%
x = train.iloc[:,1:]
y = train.iloc[:,0]
# %%
clf.fit(x,y)
# %%
prediction = clf.predict(test)
# %%
for i in range(240,250):
    d=test.iloc[i].values
    d.shape=(28,28)
    print(prediction[i])
    plt.imshow(255-d,cmap='gray')
    plt.show()
# %%
label = range(1, len(prediction) + 1)
df = pd.DataFrame(data = zip(label, prediction),columns = ['ImageId','Label'])
os.chdir(r'../submissions')
df.to_csv('tree-py.csv',index=False)