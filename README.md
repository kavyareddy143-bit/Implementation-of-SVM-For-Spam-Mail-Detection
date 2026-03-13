# Implementation-of-SVM-For-Spam-Mail-Detection

## AIM:
To write a program to implement the SVM For Spam Mail Detection.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm

1.Import the necessary python packages using import statements
2.Read the given csv file using read_csv() method and print the number of contents to be displayed using df.head().
3.Split the dataset using train_test_split.
4.Calculate Y_Pred and accuracy.
5.Print all the outputs.
6.End the Program.
## Program:
```
/*
Program to implement the SVM For Spam Mail Detection..
Developed by: Annapureddy Kavya
RegisterNumber: 212225240011 
*/
```
```
import pandas as pd
data=pd.read_csv("C:/Users/acer/Downloads/spam.csv",encoding="Windows-1252")
data.head()
data.info()
data.isnull().sum()
x=data["v1"].values
y=data["v2"].values
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)
from sklearn.feature_extraction.text import CountVectorizer
cv=CountVectorizer()
x_train=cv.fit_transform(x_train)
x_test=cv.transform(x_test)
from sklearn.svm import SVC
svc=SVC()
svc.fit(x_train,y_train)
y_pred=svc.predict(x_test)
y_pred
from sklearn import metrics
accuracy=metrics.accuracy_score(y_test,y_pred)
accuracy

```

## Output:

<img width="798" height="200" alt="image" src="https://github.com/user-attachments/assets/8c2e52e1-a25e-4983-ba82-d986f110233e" />

<img width="650" height="288" alt="image" src="https://github.com/user-attachments/assets/561101ba-763d-407f-b5a8-7ee2bdc45ee1" />

<img width="255" height="152" alt="image" src="https://github.com/user-attachments/assets/d0a84724-fa65-41f4-940b-161732fae812" />

<img width="727" height="87" alt="image" src="https://github.com/user-attachments/assets/ac6654a1-a995-4a95-8ede-343a7a786b02" />

<img width="287" height="48" alt="image" src="https://github.com/user-attachments/assets/5cf202fa-2d5d-4c5c-b483-f5f559737789" />
## Result:
Thus the program to implement the SVM For Spam Mail Detection is written and verified using python programming.
