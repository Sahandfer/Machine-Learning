import matplotlib.pyplot as plt
import numpy as np

from sklearn import datasets, linear_model

def plot_graph(formula, x_range):
    x = np.array(x_range)
    y = eval(formula)
    plt.plot(x,y)

# Loading the dataset
price = [200, 300, 200, 300, 500]
size = [50, 80, 50, 80, 100]

# Reshape the data
size = np.array(size).reshape((-1, 1))

# regression model
reg = linear_model.LinearRegression()
reg.fit(size, price)

# Testing the model
size_new = 60
price_new = (size_new * reg.coef_) + reg.intercept_
print(reg.predict([[size_new]]))

plot_graph('reg.coef_*x + reg.intercept_', range(1000, 3000))
plt.scatter(size, price, color= 'red')
plt.xlabel('Size')
plt.ylabel('Price')
plt.show()