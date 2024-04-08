# Implementation-of-Logistic-Regression-Using-Gradient-Descent

## AIM:
To write a program to implement the the Logistic Regression Using Gradient Descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm:
```
1. Import the required libraries.
2. Load the dataset.
3. Define X and Y array.
4. Define a function for costFunction,cost and gradient.
5. Define a function to plot the decision boundary. 6.Define a function to predict the 
   Regression value.
```
## Program:
```
/*
Program to implement the the Logistic Regression Using Gradient Descent.
Developed by: SHRIKRISHNA V
RegisterNumber: 212223040198 
*/
```
```C
import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize

data=np.loadtxt("ex2data1.txt",delimiter=',')
X=data[:,[0,1]]
y=data[:,2]

X[:5]

y[:5]

plt.figure()
plt.scatter(X[y==1][:,0],X[y==1][:,1],label="Admitted")
plt.scatter(X[y==0][:,0],X[y==0][:,1],label="Not Admitted")
plt.xlabel("Exam 1 score")
plt.ylabel("Exam 2 score")
plt.legend()
plt.show()

def sigmoid(z):
    return 1/(1+np.exp(-z))

plt.plot()
X_plot=np.linspace(-10,10,100)
plt.plot(X_plot,sigmoid(X_plot))
plt.show()

def costFunction (theta,X,y):
    h=sigmoid(np.dot(X,theta))
    J=-(np.dot(y,np.log(h))+np.dot(1-y,np.log(1-h)))/X.shape[0]
    grad=np.dot(X.T,h-y)/X.shape[0]
    return J,grad

X_train=np.hstack((np.ones((X.shape[0],1)),X))
theta=np.array([0,0,0])
J,grad=costFunction(theta,X_train,y)
print(J)
print(grad)

X_train=np.hstack((np.ones((X.shape[0],1)),X))
theta=np.array([-24,0.2,0.2])
J,grad=costFunction(theta,X_train,y)
print(J)
print(grad)

def cost (theta,X,y):
    h=sigmoid(np.dot(X,theta))
    J=-(np.dot(y,np.log(h))+np.dot(1-y,np.log(1-h)))/X.shape[0]
    return J

def gradient (theta,X,y):
    h=sigmoid(np.dot(X,theta))
    grad=np.dot(X.T,h-y)/X.shape[0]
    return grad

X_train=np.hstack((np.ones((X.shape[0],1)),X))
theta=np.array([0,0,0])
res=optimize.minimize(fun=cost,x0=theta,args=(X_train,y),method='Newton-CG',jac=gradient)
print(res.fun)
print(res.x)

def plotDecisionBoundary(theta,X,y):
    x_min,x_max=X[:,0].min()-1,X[:,0].max()+1
    y_min,y_max=X[:,1].min()-1,X[:,1].max()+1
    xx,yy=np.meshgrid(np.arange(x_min,x_max,0.1),np.arange(y_min,y_max,0.1))
    X_plot=np.c_[xx.ravel(),yy.ravel()]
    X_plot=np.hstack((np.ones((X_plot.shape[0],1)),X_plot))
    y_plot=np.dot(X_plot,theta).reshape(xx.shape)
    
    plt.figure()
    plt.scatter(X[y==1][:,0],X[y==1][:,1],label="Admitted")
    plt.scatter(X[y==0][:,0],X[y==0][:,1],label="Not Admitted")
    plt.contour(xx,yy,y_plot,levels=[0])
    plt.xlabel("Exam 1 score")
    plt.ylabel("Exam 2 score")
    plt.legend()
    plt.show()


plotDecisionBoundary(res.x,X,y)

prob=sigmoid(np.dot(np.array([1,45,85]),res.x))
print(prob)

def predict(theta,X):
    X_train =np.hstack((np.ones((X.shape[0],1)),X))
    prob=sigmoid(np.dot(X_train,theta))
    return (prob>=0.5).astype(int)
np.mean(predict(res.x,X)==y)
```

## Output:
## Array Value of x
![199067370-21f6e068-3851-4596-bad3-35dc02d079a6](https://github.com/charumathiramesh/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/120204455/e5311ce6-a9ee-4086-b99f-f8c6b4491790)
## Array Value of y

![199067368-23904c41-d2d1-4e62-83d0-29a65b810abe](https://github.com/charumathiramesh/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/120204455/462fb7c0-fff3-4b13-94d0-b6498c351f26)
## Exam 1 - score graph

![199067364-67b76106-9b8d-4758-a093-ec7e7f4b2d32](https://github.com/charumathiramesh/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/120204455/edc4acfc-30af-40ec-9c5e-eac35cb89e19)
## Sigmoid function graph


![199067359-63750fd2-98e8-438d-a32c-ae84cb1d27e4](https://github.com/charumathiramesh/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/120204455/9bd4bfca-0274-4d02-97ea-88bc4274d31a)
## X_train_grad value
![199067356-69f818c1-d425-48e8-beb3-281e00b6ecba](https://github.com/charumathiramesh/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/120204455/63a6de99-e789-4656-8200-b5f9cea9747b)
## Y_train_grad value

![199067352-588a14f1-b111-4fc6-801d-acb4fd847520](https://github.com/charumathiramesh/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/120204455/ced57c1b-be0d-48a9-8d21-504778656c5f)
## Print res.x
![199067351-3e334116-ed7b-441b-93e6-20737be81d24](https://github.com/charumathiramesh/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/120204455/98c18a7b-e6c0-46db-bc53-05752a2fefbd)

## Decision boundary - graph for exam score
![199067346-56d58684-54aa-478a-98ac-f9841f1b846e](https://github.com/charumathiramesh/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/120204455/d0a35897-b5a6-42a3-a856-00ea5086697f)
## Proability value
![199067342-fbdbcd76-c1d0-4fb3-95cb-d847e85e0d51](https://github.com/charumathiramesh/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/120204455/ccfd7a31-69c4-41ab-bac5-c019aa989b86)
## Prediction value of mean
![199067377-9f1bdbbb-7868-4b11-8bed-f98680735040](https://github.com/charumathiramesh/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/120204455/1d6e9ca6-3ecf-4029-828b-1993ca653c66)


## Result:
Thus the program to implement the the Logistic Regression Using Gradient Descent is written and verified using python programming.

