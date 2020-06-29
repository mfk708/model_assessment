# -*- coding: utf-8 -*-
"""
Created on Tue Jun 16 15:39:48 2020

@author: moham
"""
import pandas as pd
import numpy as np

# Import clean data 
path = 'https://s3-api.us-geo.objectstorage.softlayer.net/cf-courses-data/CognitiveClass/DA0101EN/module_5_auto.csv'
df = pd.read_csv(path)
df.head()
df.to_csv('module_5_auto.csv')
df=df._get_numeric_data() #gets just the numeric data.
df.head()
%%capture
! pip install ipywidgets
from IPython.display import display
from IPython.html import widgets 
from IPython.display import display
from ipywidgets import interact, interactive, fixed, interact_manual

def DistributionPlot(RedFunction, BlueFunction, RedName, BlueName, Title):
    width = 12
    height = 10
    plt.figure(figsize=(width, height))

    ax1 = sns.distplot(RedFunction, hist=False, color="r", label=RedName)
    ax2 = sns.distplot(BlueFunction, hist=False, color="b", label=BlueName, ax=ax1)

    plt.title(Title)
    plt.xlabel('Price (in dollars)')
    plt.ylabel('Proportion of Cars')

    plt.show()
    plt.close()
    
def PollyPlot(xtrain, xtest, y_train, y_test, lr,poly_transform):
    width = 12
    height = 10
    plt.figure(figsize=(width, height))
    
    
    #training data 
    #testing data 
    # lr:  linear regression object 
    #poly_transform:  polynomial transformation object 
 
    xmax=max([xtrain.values.max(), xtest.values.max()])

    xmin=min([xtrain.values.min(), xtest.values.min()])

    x=np.arange(xmin, xmax, 0.1)


    plt.plot(xtrain, y_train, 'ro', label='Training Data')
    plt.plot(xtest, y_test, 'go', label='Test Data')
    plt.plot(x, lr.predict(poly_transform.fit_transform(x.reshape(-1, 1))), label='Predicted Function')
    plt.ylim([-10000, 60000])
    plt.ylabel('Price')
    plt.legend()
    
#training and testing
y_data = df['price']
x_data=df.drop('price',axis=1) #drop price data in x data

from sklearn.model_selection import train_test_split
#randomly split our data into training and testing data using the function train_test_split.
x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.15, random_state=1)

print("number of test samples :", x_test.shape[0])
print("number of training samples:",x_train.shape[0])

from sklearn.linear_model import LinearRegression#import LinearRegression from the module linear_model.
lre=LinearRegression() #create a Linear Regression object
lre.fit(x_train[['horsepower']], y_train)  #fit the model using the feature horsepower
lre.score(x_test[['horsepower']], y_test) #Calculate the R^2 on the test data
lre.score(x_train[['horsepower']], y_train) ##Calculate the R^2 on the train data

#Cross-validation Score
from sklearn.model_selection import cross_val_score
Rcross = cross_val_score(lre, x_data[['horsepower']], y_data, cv=4) #input the object, the feature in this case ' horsepower', the target data (y_data). The parameter 'cv' determines the number of folds; in this case 4.
Rcross     
print("The mean of the folds are", Rcross.mean(), "and the standard deviation is" , Rcross.std())

#use the function 'cross_val_predict' to predict the output. The function splits up the data into the specified number of folds, using one fold to get a prediction while the rest of the folds are used as test data.
from sklearn.model_selection import cross_val_predict
yhat = cross_val_predict(lre,x_data[['horsepower']], y_data,cv=4)
yhat[0:5]


## Overfitting, Underfitting and Model Selection
lr = LinearRegression()
lr.fit(x_train[['horsepower', 'curb-weight', 'engine-size', 'highway-mpg']], y_train) #create 
#Multiple linear regression objects and train the model using 'horsepower', 'curb-weight', 'engine-size' and 'highway-mpg' as features.
yhat_train = lr.predict(x_train[['horsepower', 'curb-weight', 'engine-size', 'highway-mpg']]) #Prediction using test data
yhat_train[0:5]
yhat_test = lr.predict(x_test[['horsepower', 'curb-weight', 'engine-size', 'highway-mpg']]) #Prediction using test data.
yhat_test[0:5]
#perform some model evaluation using our training and testing data separately. First, import the seaborn and matplotlibb library for plotting.
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns
#examine the distribution of the predicted values of the training data.
Title = 'Distribution  Plot of  Predicted Value Using Training Data vs Training Data Distribution'
DistributionPlot(y_train, yhat_train, "Actual Values (Train)", "Predicted Values (Train)", Title)
#So far the model seems to be doing well in learning from the training dataset.
#But what happens when the model encounters new data from the testing dataset? 
#When the model generates new values from the test data, we see the distribution
#of the predicted values is much different from the actual target values.
Title='Distribution  Plot of  Predicted Value Using Test Data vs Data Distribution of Test Data'
DistributionPlot(y_test,yhat_test,"Actual Values (Test)","Predicted Values (Test)",Title)  
#Let's see if polynomial regression also exhibits a drop in the prediction accuracy when
#analysing the test dataset.
from sklearn.preprocessing import PolynomialFeatures
#Overfitting: Overfitting occurs when the model fits the noise, not the underlying process.
#use 55 percent of the data for testing and the rest for training:
x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.45, random_state=0)
#perform a degree 5 polynomial transformation on the feature 'horse power':
pr = PolynomialFeatures(degree=5)
x_train_pr = pr.fit_transform(x_train[['horsepower']])
x_test_pr = pr.fit_transform(x_test[['horsepower']])
pr
#Now let's create a linear regression model "poly" and train it:
poly = LinearRegression()
poly.fit(x_train_pr, y_train)
# We can see the output of our model using the method "predict." 
#then assign the values to "yhat".
yhat = poly.predict(x_test_pr)
yhat[0:5]
#Let's take the first five predicted values and compare it to the actual targets.
print("Predicted values:", yhat[0:4])
print("True values:", y_test[0:4].values)
#Now use the function "PollyPlot" that we defined at the beginning of the lab to display the training data, testing data, and the predicted function.
PollyPlot(x_train[['horsepower']], x_test[['horsepower']], y_train, y_test, poly,pr)
#the estimated function appears to track the data but around 200 horsepower, the function begins to diverge from the data points
#R^2 of the training data
poly.score(x_train_pr, y_train)
#R^2 of the test data:
poly.score(x_test_pr, y_test) #Negative R^2 is a sign of overfitting.
#Now, Let's see how the R^2 changes on the test data for different order polynomials and plot the results:
Rsqu_test = []

order = [1, 2, 3, 4]
for n in order:
    pr = PolynomialFeatures(degree=n)
    
    x_train_pr = pr.fit_transform(x_train[['horsepower']])
    
    x_test_pr = pr.fit_transform(x_test[['horsepower']])    
    
    lr.fit(x_train_pr, y_train)
    
    Rsqu_test.append(lr.score(x_test_pr, y_test))

plt.plot(order, Rsqu_test)
plt.xlabel('order')
plt.ylabel('R^2')
plt.title('R^2 Using Test Data')
plt.text(3, 0.75, 'Maximum R^2 ')
#We see the R^2 gradually increases until an order three polynomial is used. 
#Then the  R^2 dramatically decreases at four.
##Ridge regression
#review Ridge Regression we will see how the parameter Alfa changes the model. 
#Just a note here our test data will be used as validation data.
#Let's perform a degree two polynomial transformation on our data.
pr=PolynomialFeatures(degree=2)
x_train_pr=pr.fit_transform(x_train[['horsepower', 'curb-weight', 'engine-size', 'highway-mpg','normalized-losses','symboling']])
x_test_pr=pr.fit_transform(x_test[['horsepower', 'curb-weight', 'engine-size', 'highway-mpg','normalized-losses','symboling']])
#Let's import Ridge from the module linear models.
from sklearn.linear_model import Ridge
#Let's create a Ridge regression object, setting the regularization parameter to 0.1 
RigeModel=Ridge(alpha=0.1)
#Like regular regression, you can fit the model using the method <b>fit</b>.
RigeModel.fit(x_train_pr, y_train)
#Similarly, you can obtain a prediction:
yhat = RigeModel.predict(x_test_pr)
#Let's compare the first five predicted samples to our test set:
print('predicted:', yhat[0:4])
print('test set :', y_test[0:4].values)
#We select the value of Alfa that minimizes the test error, for example, we can use a for loop:
Rsqu_test = []
Rsqu_train = []
dummy1 = []
ALFA = 10 * np.array(range(0,1000))
print(ALFA)
for alfa in ALFA:
    RigeModel = Ridge(alpha=alfa) 
    RigeModel.fit(x_train_pr, y_train)
    Rsqu_test.append(RigeModel.score(x_test_pr, y_test))
    Rsqu_train.append(RigeModel.score(x_train_pr, y_train))
#plot out the value of R^2 for different Alphas
width = 12
height = 10
plt.figure(figsize=(width, height))
plt.plot(ALFA,Rsqu_test, label='validation data  ')
plt.plot(ALFA,Rsqu_train, 'r', label='training Data ')
plt.xlabel('alpha')
plt.ylabel('R^2')
plt.legend()
#The term Alfa is a hyperparameter, sklearn has the class GridSearchCV to make the process of 
#finding the best hyperparameter simpler
#Let's import GridSearchCV from the module model_selection.
from sklearn.model_selection import GridSearchCV
#create a dictionary of parameter values:
parameters1= [{'alpha': [0.001,0.1,1, 10, 100, 1000, 10000, 100000, 100000]}]
parameters1
#Create a ridge regions object:
RR=Ridge()
RR
#Create a ridge grid search object:
Grid1 = GridSearchCV(RR, parameters1,cv=4)
#Fit the model
Grid1.fit(x_data[['horsepower', 'curb-weight', 'engine-size', 'highway-mpg']], y_data)
#The object finds the best parameter values on the validation data. We can obtain the estimator 
#with the best parameters and assign it to the variable BestRR as follows:
BestRR=Grid1.best_estimator_
BestRR
#We now test our model on the test data:
BestRR.score(x_test[['horsepower', 'curb-weight', 'engine-size', 'highway-mpg']], y_test)   
