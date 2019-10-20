import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from utils import get_data

train, test = get_data()
# %%
clf = DecisionTreeClassifier()
# %%
xtrain = train[0:21000,1:]
train_label = train[0:21000,0]
# %%
clf.fit(xtrain,train_label)
# %%
xtest = train[21000:,1:]
actual_label = train[21000:,0]
d=xtest[8]
# %%
d.shape=(28,28)
plt.imshow(255-d,cmap='gray')
print(clf.predict([xtest[8]]))
plt.show()
# %%
p = clf.predict(xtest)
# %%
count=0
for i in range(21000):
    count +=1 if p[i]==actual_label[i] else 0
print(count/21000)