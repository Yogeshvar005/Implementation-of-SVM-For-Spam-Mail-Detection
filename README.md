# Implementation-of-SVM-For-Spam-Mail-Detection

## AIM:
To write a program to implement the SVM For Spam Mail Detection.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Import the necessary python packages using import statements.

2.Read the given csv file using read_csv() method and print the number of contents to be displayed using df.head().

3.Split the dataset using train_test_split.

4.Calculate Y_Pred and accuracy.

5.Print all the outputs.

6.End the Program.
## PROGRAM :
```python
'''
Program to implement the SVM For Spam Mail Detection..
Developed by : YOGESHVAR M
RegisterNumber : 212222230180
'''
import chardet
file='spam.csv'
with open (file,'rb') as rawdata:
    result = chardet.detect(rawdata.read(100000))
result

import pandas as pd
data=pd.read_csv("spam.csv",encoding='windows-1252')

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

## OUTPUT :
### Encoding
![image](https://github.com/Abburehan/Implementation-of-SVM-For-Spam-Mail-Detection/assets/138849336/881fe7fe-23d5-4ed9-bdfa-139923432f8a)

### Head()
![image](https://github.com/Abburehan/Implementation-of-SVM-For-Spam-Mail-Detection/assets/138849336/c62b6cb2-1259-4f2b-9b55-01ab49fae977)

### Info()
![image](https://github.com/Abburehan/Implementation-of-SVM-For-Spam-Mail-Detection/assets/138849336/fe7dc709-82b5-4be0-babc-9223c4525bb5)

### isnull().sum()
![image](https://github.com/Abburehan/Implementation-of-SVM-For-Spam-Mail-Detection/assets/138849336/c7c4c3a1-90d9-4b25-b514-99fdababf256)

### Prediction of y
![image](https://github.com/Abburehan/Implementation-of-SVM-For-Spam-Mail-Detection/assets/138849336/067f7cf1-3e4f-4d43-a539-ecde5276b695)

### Accuracy
![image](https://github.com/Abburehan/Implementation-of-SVM-For-Spam-Mail-Detection/assets/138849336/130a0281-9bce-450c-90b2-f65f76e51260)

## Result:
Thus the program to implement the SVM For Spam Mail Detection is written and verified using python programming.
