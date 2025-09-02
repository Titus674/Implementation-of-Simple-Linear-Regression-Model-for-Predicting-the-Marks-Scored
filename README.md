# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Get the independent variable X and dependent variable Y.
2. Calculate the mean of the X -values and the mean of the Y -values.
3. Find the slope m of the line of best fit using the formula. 
<img width="231" alt="image" src="https://user-images.githubusercontent.com/93026020/192078527-b3b5ee3e-992f-46c4-865b-3b7ce4ac54ad.png">
4. Compute the y -intercept of the line by using the formula:
<img width="148" alt="image" src="https://user-images.githubusercontent.com/93026020/192078545-79d70b90-7e9d-4b85-9f8b-9d7548a4c5a4.png">
5. Use the slope m and the y -intercept to form the equation of the line.
6. Obtain the straight line equation Y=mX+b and plot the scatterplot.

## Program:
```
/*
Program to implement the simple linear regression model for predicting the marks scored.
Developed by: Titus Ratna Kumar Karivella 
RegisterNumber:  212224230292
*/
```
```python 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
#import libraries to find mae, mse
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error

#read csv file
df= pd.read_csv('/content/student_scores.csv')

#displaying the content in datafile
df.head()
df.tail()

# Segregating data to variables
X=df.iloc[:,:-1].values
X
y=df.iloc[:,-1].values
y

#splitting train and test data
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=1/2,random_state=0)

#import linear regression model and fit the model with the data
from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(X_train,y_train)

#displaying predicted values
y_pred=regressor.predict(X_test)
y_pred

#displaying actual values
y_test

# graph plot for training data

plt.scatter(X_train, y_train, color='red')
plt.plot(X_train, regressor.predict(X_train), color='blue')
plt.title("Hours vs Scores (Training Set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show() # This command displays the plot

# graph plot for test data
plt.scatter(X_test, y_test, color='red')
plt.plot(X_test, regressor.predict(X_test), color='blue')
plt.title("Hours vs Scores (Testing Set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show() # This command displays the plot

#find mae,mse,rmse
mse=mean_squared_error(y_test,y_pred)
print('MSE = ',mse)
mae=mean_absolute_error(y_test,y_pred)
print('MAE = ',mae)
rmse=np.sqrt(mse)
print('RMSE = ',rmse)
```


## Output:
### HEAD VALUES

<img width="282" height="215" alt="Screenshot 2025-08-23 140840" src="https://github.com/user-attachments/assets/2ec1537e-8b89-4825-bddd-f0c278901726" />

### TAIL VALUES

<img width="189" height="213" alt="Screenshot 2025-08-23 140859" src="https://github.com/user-attachments/assets/8e958e0c-8418-49a4-8935-9a764c0b861b" />

### X VALUES

<img width="210" height="502" alt="Screenshot 2025-08-23 141059" src="https://github.com/user-attachments/assets/29b0ae14-743a-46f3-b42e-1d42e80aee33" />

### Y VALUES 

<img width="709" height="42" alt="Screenshot 2025-08-23 141333" src="https://github.com/user-attachments/assets/52af8f71-7a69-4222-bf26-ab60d2243358" />


### PREDICTED VALUES

<img width="711" height="89" alt="Screenshot 2025-08-23 140659" src="https://github.com/user-attachments/assets/7b156aff-4f41-4768-864c-bb16f587f121" />


### ACTUAL VALUES 

<img width="594" height="46" alt="Screenshot 2025-08-23 140652" src="https://github.com/user-attachments/assets/0e0dcfc5-a190-4d7b-881f-c1490307becb" />

### TRAINING SET

<img width="588" height="421" alt="Screenshot 2025-08-23 140436" src="https://github.com/user-attachments/assets/c5ca1120-3d75-46a5-93a8-01177beb9abd" />

### TESTING SET

<img width="573" height="431" alt="Screenshot 2025-08-23 141442" src="https://github.com/user-attachments/assets/107a882e-e004-4a28-8e2c-939769cc8440" />

### MSE, MAE and RMSE

<img width="194" height="69" alt="Screenshot 2025-08-23 140641" src="https://github.com/user-attachments/assets/9dc7f088-dcd0-4a2a-9f53-cd9e1cd4c3fb" />


## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
