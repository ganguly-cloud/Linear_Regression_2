import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

''' FROM SCRACH '''
''' ===========  '''

data = pd.read_csv('headbrain.csv')
print data.shape  # (237, 4)
print data.head()

x = data['Head Size(cm^3)'].values
y = data['Brain Weight(grams)'].values
print x[:5]
print y[:5]

# Calculate m and c

mean_x =np.mean(x)
mean_y =np.mean(y)
print mean_x

# total no of values
n = len(x)
print n  # 237
# Using the formula to calculate b1 and b2

num =0
denom =0
'''formula for m 
m=sum(x-mean_x)*(y-mean_y) / sum(x-mean_x)^2 or **2'''
for i in range(n):
    num +=(x[i] - mean_x)*(y[i] - mean_y)
    denom +=(x[i] - mean_x) **2

m = num /denom
c = mean_y -(m * mean_x)  # formula for c =mean_y - ( m* mean_x )
# print co efficients
print m,c    # 0.26342933948939945 325.57342104944223

# plotting values and regression line

max_x = np.max(x) 
min_x = np.min(x) 
print max_x   # 4847
print min_x   # 2620
# calculate line values x and y

X = np.linspace(min_x,max_x,1000)
Y = m*x+c

print X[:6]
'''
[2620.         2622.22922923 2624.45845846 2626.68768769 2628.91691692
 2631.14614615] '''
print y[:6]
'''
[1015.75829051 1016.3455349  1016.93277928 1017.52002366 1018.10726805
 1018.69451243] '''

# now ploting
# for plotting linear OR predicted line
plt.plot(x,Y,color ='blue',label ='Regression Line')
# plot scatter plot for input values
plt.scatter(x,y, c='r',label ='Scatter plot')

plt.xlabel('Head size in cm3')
plt.ylabel('brain width in grams')
plt.legend()
plt.savefig('input_output_after_pred')
plt.show()

# have to find how good our model is R2 method or RMSE method
# Here we are using R2_method
ss_t =0
ss_r =0
for i in range(n):
    y_pred = m*x[i] +c
    ss_t +=(y[i] - mean_y)**2
    ss_r +=(y[i] - y_pred)**2
r2 = 1-(ss_r /ss_t)
print r2    #  0.6393117199570003
    

""" using sklearn Library """

from sklearn.linear_model import LinearRegression
from sklearn.linear_model.LinearRegression
from sklearn.metrics import mean_squared_error

x = x.reshape(-1,1)
print x

reg = LinearRegression()

reg = reg.fit(x,y)

y_predict = reg.predict(x)

print y_predict[:6]

mse = mean_squared_error (Y,y_pred)

rmse = np.sqrt(mse)

r2_score =reg.score(x,y)

print r2_score   #  0.639311719957

