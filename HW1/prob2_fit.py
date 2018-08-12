'''
Assignment 1: Problem 2 - Polynomial Regression
Submission by: Anand Gopalakrishnan (aug440@psu.edu)
'''

import os  
import numpy as np  
import pandas as pd  
import matplotlib.pyplot as plt  

'''
IST 597: Foundations of Deep Learning
Problem 2: Polynomial Regression & 

@author - Alexander G. Ororbia II

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.
'''

# NOTE: you will need to tinker with the meta-parameters below yourself (do not think of them as defaults by any means)
# meta-parameters for program
trial_name = 'p1_fit' # will add a unique sub-string to output of this program
degree = 15 # p, order of model
beta = 0.1 # regularization coefficient
alpha = 1.0 # step size coefficient
eps = 0.000001   #controls convergence criterion
n_epoch = 50000 # number of epochs (full passes through the dataset)

# begin simulation
def regress(X, theta):
	# WRITEME: write your code here to complete the routine
	y_hat = np.dot(X,theta[1].T) + theta[0]
	return y_hat

def gaussian_log_likelihood(mu, y):
	# WRITEME: write your code here to complete the routine
	mse = np.sum(np.power(mu - y,2))
	return mse
	
def computeCost(X, y, theta, beta): # loss is now Bernoulli cross-entropy/log likelihood
	# WRITEME: write your code here to complete the routine
	y_hat = regress(X,theta)
	mse = gaussian_log_likelihood(y_hat,y)
	N = 1/float(2*X.shape[0])
	
	# l2 penalty term
	l2_pen = beta*N*np.sum(np.power(theta[0],2))
	return mse*N + l2_pen
	
def computeGrad(X, y, theta, beta):
	# WRITEME: write your code here to complete the routine (
	# NOTE: you do not have to use the partial derivative symbols below, they are there to guide your thinking)
	y_hat = regress(X,theta)
	M = 1/float(X.shape[0])
	
	dL_dfy = 1 # derivative w.r.t. to model output units (fy)
	dL_db = M*np.sum(y_hat - y, axis=0) # derivative w.r.t. model weights b
	dL_dw = M*(np.dot(((y_hat - y).T),X) + np.sum(beta*theta[1], axis=0)) # derivative w.r.t model bias w
	nabla = (dL_db, dL_dw) # nabla represents the full gradient
	return nabla

path = os.getcwd() + '/data/prob2.dat'  
data = pd.read_csv(path, header=None, names=['X', 'Y']) 

# set X (training data) and y (target variable)
cols = data.shape[1]  
X = data.iloc[:,0:cols-1]  
y = data.iloc[:,cols-1:cols] 

# convert from data frames to numpy matrices
X = np.array(X.values)
y = np.array(y.values)

# apply feature map to input features x1
# WRITEME: write code to turn X_feat into a polynomial feature map (hint: you could use a loop and array concatenation)
# feature mapping function
def feature_map(X, degree):
	X_feature = np.zeros((X.shape[0], degree))
	# loop through all samples
	for i in range(X.shape[0]):
		
		# loop through all polynomial degrees
		for j in range(0,degree):
			X_feature[i,j] = np.power(X[i],j+1)
	return X_feature

# applying feature mapping to input features x_1
X_feature = feature_map(X, degree)

# convert to numpy arrays and initalize the parameter array theta 
w = np.zeros((1,X_feature.shape[1]))
b = np.array([0])
theta = (b, w)

L = computeCost(X_feature, y, theta, beta)
print("-1 L = {0}".format(L))
i = 0
cost = [0] #you can use this list variable to help you create the loss versus epoch plot at the end (if you want)
while(i < n_epoch):
	dL_db, dL_dw = computeGrad(X_feature, y, theta, beta)
	b = theta[0]
	w = theta[1]
	# update rules go here...
	# WRITEME: write your code here to perform a step of gradient descent & record anything else desired for later
	b = b - alpha*dL_db
	w = w - alpha*dL_dw
	theta = (b,w) #overwriting theta
	L = computeCost(X_feature, y, theta, beta)
	cost.append(L) # tracking cost at end of each epoch
	# WRITEME: write code to perform a check for convergence (or simply to halt early)
	recent_cost= cost[-2:]
	if(np.abs(recent_cost[1] - recent_cost[0]) < eps):
		print ("Training terminated at epoch", i)
		break
	
	print(" {0} L = {1}".format(i,L))
	i += 1
# print parameter values found after the search
print("w = ",w)
print("b = ",b)

kludge = 0.25
# visualize the fit against the data
X_test = np.linspace(data.X.min(), data.X.max(), 100)
X_feat = np.expand_dims(X_test, axis=1) # we need this otherwise, the dimension is missing (turns shape(value,) to shape(value,value))

# apply feature map to input features x1
# WRITEME: write code to turn X_feat into a polynomial feature map (hint: you could use a loop and array concatenation)
X_test_feature = feature_map(X_feat, degree)

plt.figure()
plt.plot(X_test, regress(X_test_feature, theta), label="Model")
plt.scatter(X[:,0], y, edgecolor='g', s=20, label="Samples")
plt.xlabel("x")
plt.ylabel("y")
plt.title("order 15") # label plots tested with different beta
plt.xlim((np.amin(X_test) - kludge, np.amax(X_test) + kludge))
plt.ylim((np.amin(y) - kludge, np.amax(y) + kludge))
plt.legend(loc="best")
# WRITEME: write your code here to save plot to disk (look up documentation/inter-webs for matplotlib)
#plt.savefig('prob_2_poly_regress.png')


plt.figure()
plt.plot(cost)
plt.xlabel("epochs")
plt.ylabel("cost")
plt.title("cost vs epochs")
#plt.savefig('prob_3_cost_epochs.png')
#plt.show()
