'''
Assignment 1: Problem 3 - Logistic Regression
Submission by: Anand Gopalakrishnan (aug440@psu.edu)
'''
import os  
import numpy as np  
import pandas as pd  
import matplotlib.pyplot as plt  

'''
IST 597: Foundations of Deep Learning
Problem 3: Multivariate Regression & Classification

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
trial_name = 'p6_reg0' # will add a unique sub-string to output of this program
degree = 6 # p, degree of model (LEAVE THIS FIXED TO p = 6 FOR THIS PROBLEM)
beta = 0.1 # regularization coefficient
alpha = 4.5 # step size coefficient
n_epoch = 500 # number of epochs (full passes through the dataset)
eps = 0.0 # controls convergence criterion
threshold = 0.9 # threshold of logistic function
# begin simulation

def sigmoid(z):
	# WRITEME: write your code here to complete the routine
	phi_z = np.zeros((z.shape[0],1))
	phi_z = 1 / (1 + np.exp(-z))
	return phi_z

def regress(X, theta):
	# WRITEME: write your code here to complete the routine
	z = np.dot(X, theta[1].T) + theta[0]
	z = sigmoid(z)
	return z

def predict(X, theta):  
	# WRITEME: write your code here to complete the routine
	z = regress(X, theta)
	p = np.array(z>=threshold, dtype=int)
	return p	

def bernoulli_log_likelihood(phi_z, y):
	# WRITEME: write your code here to complete the routine
	log_likelihood = -np.dot(y.T, np.log(phi_z)) - np.dot((1 - y).T, np.log(1 - phi_z))
	
	return log_likelihood
	
def computeCost(X, y, theta, beta): # loss is now Bernoulli cross-entropy/log likelihood
	# WRITEME: write your code here to complete the routine
	z = regress(X,theta)
	log_likelihood = bernoulli_log_likelihood(z,y)
	N = 1/float(2*X.shape[0])
	
	# l2 penalty term
	l2_pen = beta*N*np.sum(theta[1]**2, axis=1)
	
	return log_likelihood*2*N + l2_pen
	
def computeGrad(X, y, theta, beta): 
	# WRITEME: write your code here to complete the routine (
	# NOTE: you do not have to use the partial derivative symbols below, they are there to guide your thinking)
	y_hat = regress(X,theta)
	M = 1/float(X.shape[0])
	
	dL_dfy = 1 # derivative w.r.t. to model output units (fy)
	dL_db = M*np.sum((y_hat - y), axis=0) # derivative w.r.t. model weights b
	dL_dw = M*(np.dot(((y_hat - y).T),X) + np.sum(beta*theta[1], axis=0)) # derivative w.r.t model bias w
	nabla = (dL_db, dL_dw) # nabla represents the full gradient
	return nabla
	
path = os.getcwd() + '/data/prob3.dat'  
data2 = pd.read_csv(path, header=None, names=['Test 1', 'Test 2', 'Accepted'])

positive = data2[data2['Accepted'].isin([1])]  
negative = data2[data2['Accepted'].isin([0])]
 
x1 = data2['Test 1']  
x2 = data2['Test 2']

# apply feature map to input features x1 and x2
cnt = 0
for i in range(1, degree+1):  
	for j in range(0, i+1):
		data2['F' + str(i) + str(j)] = np.power(x1, i-j) * np.power(x2, j)
		cnt += 1

data2.drop('Test 1', axis=1, inplace=True)  
data2.drop('Test 2', axis=1, inplace=True)

# set X and y
cols = data2.shape[1]  
X2 = data2.iloc[:,1:cols]  
y2 = data2.iloc[:,0:1]

# convert to numpy arrays and initalize the parameter array theta
X2 = np.array(X2.values)  
y2 = np.array(y2.values)  
w = np.random.random((1,X2.shape[1]))
b = np.array([0])
theta = (b, w)

L = computeCost(X2, y2, theta, beta)
print("-1 L = {0}".format(L))
i = 0
cost = [0] # you can use this list variable to help you create the loss versus epoch plot at the end (if you want)
halt = 0
while(i < n_epoch and halt == 0):
	dL_db, dL_dw = computeGrad(X2, y2, theta, beta)
	b = theta[0]
	w = theta[1]
	# update rules go here...
	# WRITEME: write your code here to perform a step of gradient descent & record anything else desired for later
	b = b - alpha*dL_db
	w = w - alpha*dL_dw
	theta = (b,w) #overwriting theta

	L = computeCost(X2, y2, theta, beta)
	cost.append(L)
	# WRITEME: write code to perform a check for convergence (or simply to halt early)
	recent_cost= cost[-2:]
	if(i>0 and np.abs(recent_cost[1] - recent_cost[0]) < eps):
		print ("Training terminated at epoch", i)
		halt = 1
		
	print(" {0} L = {1}".format(i,L))
	i += 1
# print parameter values found after the search
print("w = ",w)
print("b = ",b)

predictions = predict(X2, theta)

# compute error (100 - accuracy)
err = np.sum(np.abs(predictions - y2)) / float(y2.shape[0])
# WRITEME: write your code here calculate your actual classification error (using the "predictions" variable)
print 'Error = {0}%'.format(err * 100.)


# make contour plot
xx, yy = np.mgrid[-1.2:1.2:.01, -1.2:1.2:.01]
xx1 = xx.ravel()
yy1 = yy.ravel()
grid = np.c_[xx1, yy1]
grid_nl = []
# re-apply feature map to inputs x1 & x2
for i in range(1, degree+1):  
	for j in range(0, i+1):
		feat = np.power(xx1, i-j) * np.power(yy1, j)
		if (len(grid_nl) > 0):
			grid_nl = np.c_[grid_nl, feat]
		else:
			grid_nl = feat
probs = regress(grid_nl, theta).reshape(xx.shape)

f, ax = plt.subplots(figsize=(8, 6))
ax.contour(xx, yy, probs, levels=[threshold], cmap="Greys", vmin=0, vmax=.6)

ax.scatter(x1, x2, c=y2, s=50,
           cmap="RdBu", vmin=-.2, vmax=1.2,
           edgecolor="white", linewidth=1)

ax.set(aspect="equal",
       xlim=(-1.5, 1.5), ylim=(-1.5, 1.5),
       xlabel="$X_1$", ylabel="$X_2$")
plt.title("beta = 0.1")
# WRITEME: write your code here to save plot to disk (look up documentation/inter-webs for matplotlib)
#plt.savefig('prob_2_logistic_decision_boundary.png')

# plotting cost vs epochs
# removing zero value at the beginning of cost list
#cost = cost[1:]
plt.figure()
plt.plot(cost)
plt.xlabel("epochs")
plt.ylabel("cost")
plt.title("cost vs epochs")
#plt.savefig('prob_3_cost_epochs.png')
#plt.show()