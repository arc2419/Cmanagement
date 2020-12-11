import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pickle

df = pd.read_csv(r'C:\Users\Abha\Documents\Store_visit_test.csv')

array = df.values
X = array[:,[0,1]]
y = array[:,2]


from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(X, y)

pickle.dump(model, open('model.pkl','wb'))

model = pickle.load(open('model.pkl','rb'))
print(model.predict([[4, 300, 500]]))