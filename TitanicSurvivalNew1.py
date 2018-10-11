# -*- coding: utf-8 -*-
"""
Created on Fri Oct  5 23:08:55 2018

@author: Monil
"""

#%% Titanic train data
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

titanic_train=pd.read_excel('../input/Titanic_Survival_Train.xls')
titanic_test=pd.read_excel('../input/Titanic_Survival_Test.xls')
print(titanic_train)
print(titanic_test)
#%%
print(titanic_train.columns.values)
print(titanic_test.columns.values)
titanic_train.shape
titanic_test.shape
#%% checking head
titanic_train.head(5)
titanic_test.head(5)
#%% checking tail
titanic_train.tail(5)
titanic_test.tail(5)
#%%datatypes
titanic_train.dtypes
titanic_test.dtypes
#%%
titanic_train.describe(include='all')
#%%
titanic_train.Age.describe()
#%% Male and Female total and how many survived
from PIL import Image
jpgfile = Image.open('../input/1.jpg')
jpgfile.show()
#%%Number of People Embarked from Cherbourg(C), Queenstown(Q) and Southampton(S) 
#and Number of People who survived and breakdown sex wise.

jpgfile = Image.open('../input/2.jpg')
jpgfile.show()
#%% Agewise breakdown of people who survived
jpgfile = Image.open('../input/3.jpg')
jpgfile.show()
#%%
count_embarked = pd.value_counts(titanic_train['Embarked'], sort = True).sort_index()
count_embarked.plot(kind = 'bar')
plt.title("Embarked")
plt.xlabel("Port")
plt.ylabel("Frequency")

#%% Pasengers Class wise
count_classes = pd.value_counts(titanic_train['Pclass'], sort = True).sort_index()
count_classes.plot(kind = 'bar')
plt.title("Passenger Class")
plt.xlabel("Class")
plt.ylabel("Frequency")

#%%TOTAL VALUES FOR AGE
print(titanic_train['Age'].count())
#%%TOTAL VALUES MISSING FOR AGE
print(titanic_train['Age'].isnull().sum())
#%%TOTAL VALUES AVAILABLE FOR AGE
print(titanic_train['Age'].count() - titanic_train['Age'].isnull().sum())
age_available=titanic_train['Age'].count() - titanic_train['Age'].isnull().sum()
age_available

#%% Discretization and binning
PassengerAge=titanic_train['Age']
PassengerAge=PassengerAge.dropna()
Bins=[0,18,60,PassengerAge.max()+1]
Binlabels=['Children','Adult','Senior']
categories=pd.cut(PassengerAge,Bins,labels=Binlabels, right=False,include_lowest=True)
print(categories.value_counts())
print(categories)
#%% Passengers who have age more than 60 - Male and Female combined

df_senior=titanic_train[titanic_train['Age']>60]
df_senior.shape
df_senior['Age'].count()
#%%Passengers who have age more than 60 and are male
df_senior=titanic_train[titanic_train['Age']>60]
are_male_senior=df_senior[df_senior['Sex']=="male"]
are_male_senior.shape
are_male_senior['Age'].count()
#%%Passengers who have age more than 60 are male and who survived
male_senior_survived=titanic_train[(titanic_train['Age']>60)&(titanic_train['Sex']=='male')
&(titanic_train['Survived']==1)]
male_senior_survived.shape
male_senior_survived['Age'].count()
#%%Passengers who have age more than 60 and are female
df_senior=titanic_train[titanic_train['Age']>60]
are_female_senior=df_senior[df_senior['Sex']=="female"]
are_female_senior.shape
are_female_senior['Age'].count()
#%%Passengers who have age more than 60 are female and who survived
female_senior_survived=titanic_train[(titanic_train['Age']>60)&(titanic_train['Sex']=='female')
&(titanic_train['Survived']==1)]
female_senior_survived.shape
female_senior_survived['Age'].count()

#%%
df_senior['Age'].count()
are_male_senior['Age'].count()
male_senior_survived['Age'].count()
are_female_senior['Age'].count()
female_senior_survived['Age'].count()

print(df_senior.Age.value_counts(normalize=True).plot(kind="bar",alpha=0.8))

print(are_male_senior.Age.value_counts(normalize=True).plot(kind="bar",alpha=0.8))
print(male_senior_survived.Age.value_counts(normalize=True).plot(kind="bar",alpha=0.8))
print(are_female_senior.Age.value_counts(normalize=True).plot(kind="bar",alpha=0.8))
print(female_senior_survived.Age.value_counts(normalize=True).plot(kind="bar",alpha=0.8))

print(titanic_train.Survived.value_counts(normalize=True).plot(kind="bar",alpha=0.8))

#%% Passengers who have age less than 18 - Male and Female combined

df_child=titanic_train[titanic_train['Age']<18]
df_child.shape
df_child['Age'].count()

#%%Passengers who have age less than 18 and are male
df_child=titanic_train[titanic_train['Age']<18]
are_male_child=df_child[df_child['Sex']=="male"]
are_male_child.shape
are_male_child['Age'].count()
#%%Passengers who have age less than 18 are male and who survived
male_child_survived=titanic_train[(titanic_train['Age']<18)&(titanic_train['Sex']=='male')
&(titanic_train['Survived']==1)]
male_child_survived.shape
male_child_survived['Age'].count()
#%%Passengers who have age less than 18 and are female
df_child=titanic_train[titanic_train['Age']<18]
are_female_child=df_child[df_child['Sex']=="female"]
are_female_child.shape
are_female_child['Age'].count()
#%%Passengers who have age less than 18 are female and who survived
female_child_survived=titanic_train[(titanic_train['Age']<18)&(titanic_train['Sex']=='female')
&(titanic_train['Survived']==1)]
female_child_survived.shape
female_child_survived['Age'].count()

#%%
df_child['Age'].count()
are_male_child['Age'].count()
male_child_survived['Age'].count()
are_female_child['Age'].count()
female_child_survived['Age'].count()
#%%
print(df_child.Age.value_counts(normalize=True).plot(kind="bar",alpha=0.8))
#%%
print(are_male_child.Age.value_counts(normalize=True).plot(kind="bar",alpha=0.8))
#%%
print(male_child_survived.Age.value_counts(normalize=True).plot(kind="bar",alpha=0.8))
#%%
print(are_female_child.Age.value_counts(normalize=True).plot(kind="bar",alpha=0.8))
#%%
print(female_child_survived.Age.value_counts(normalize=True).plot(kind="bar",alpha=0.8))
#%%tabulation
#print(titanic_train)
myseries=titanic_train['Pclass']
print(myseries.value_counts()) 
type(myseries)
#%% cross table between 2 variables
pd.crosstab(titanic_train['Sex'], titanic_train['Pclass'])
#%%
#sex wise survival rate
Sexwise_survived=titanic_train["Sex"][titanic_train["Survived"] == 1].value_counts()
print (Sexwise_survived.plot(kind="bar",alpha=0.8))


#%%Treating Missing Values
print(titanic_train.isnull().sum())

#%%handling missing values in Cabin Column
#dropped Cabin column
titanic_train=titanic_train.drop('Cabin', axis=1)
titanic_train.head
#%% checking again
print(titanic_train.isnull().sum())
#%% handling missing values in Embarked Column categorical column
#filled with mode value

titanic_train['Embarked'].fillna(titanic_train['Embarked'].mode()[0],inplace=True)
#%%
titanic_train.head
#%%
print(titanic_train.isnull().sum())
#%%
print(titanic_train)
#%%handling missing values in Age Column
age_mean=titanic_train.Age.mean()
print(age_mean)
titanic_train['Age'].fillna(age_mean,inplace=True)
titanic_train.head(275)
#%%
print(titanic_train.isnull().sum())

my_df=titanic_train[["Sex","Pclass","Age"]]
my_df
#%% Dropping the issue_d column
titanic_train=titanic_train.drop(['Ticket',], axis=1)
#%% Dropping the issue_d column
titanic_test=titanic_test.drop(['Ticket'], axis=1)

#%% for preprocessing the data - creating categorical data in numerical form ,'term', 'grade',  'verification_status'
from sklearn import preprocessing
colname=['Name','Sex','Embarked']

le={}
    
for x in colname:
        le[x]=preprocessing.LabelEncoder()
for x in colname:
        titanic_train[x]=le[x].fit_transform(titanic_train.__getattr__(x))
    
#%% for preprocessing the data - creating categorical data in numerical form
from sklearn import preprocessing
colname=['Name','Sex','Embarked']

le={}
    
for x in colname:
        le[x]=preprocessing.LabelEncoder()
for x in colname:
        titanic_test[x]=le[x].fit_transform(titanic_test.__getattr__(x))
    

#%%
X_train = titanic_train.drop("Survived", axis=1)
Y_train = titanic_train["Survived"]
X_test  = titanic_test.drop("PassengerId", axis=1).copy()
X_train.shape, Y_train.shape, X_test.shape
#%%create x and y     #all variables except last one
        
X_train=titanic_train.values[:,:-1] 
Y_train=titanic_train.values[:,-1] #subset rows , subset columns 
#Y_train=Y_train.astype(int) #sometimes y is treated as object so typecast y as int

X_test=titanic_train.values[:,:-1]
y_test=titanic_train.values[:,-1] #subset rows , subset columns 
#y_test=y_test.astype(int) #sometimes y is treated as object so typecast y as int
     
#%% scale data train
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(X_train)
X_train=scaler.transform(X_train)
print(X_train)
   
#%% scale data test
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(X_test)
X_test=scaler.transform(X_test)
print(X_test)     
#print(titanic_train)

# Logistic Regression
from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression()
#fitting training data to the model
logreg.fit(X_train, Y_train)
Y_pred = logreg.predict(X_test)
print(list(zip(y_test, Y_pred)))
acc_log = round(logreg.score(X_train, Y_train) * 100, 2)
print("Accuracy of the model Logistic Regression: ", acc_log)

# Support Vector Machines
from sklearn import svm
svc_model=svm.SVC(kernel='rbf', C=1.0, gamma=0.1)
svc_model.fit(X_train, Y_train) #training the model
Y_pred = svc_model.predict(X_test)
acc_svc = round(svc_model.score(X_train, Y_train) * 100, 2)
print("Accuracy of the model SVM: ", acc_svc)

# Predicting using the Decision Tree classifier
from sklearn.tree import DecisionTreeClassifier
decision_tree = DecisionTreeClassifier()
decision_tree.fit(X_train, Y_train)
Y_pred = decision_tree.predict(X_test)
acc_decision_tree = round(decision_tree.score(X_train, Y_train) * 100, 2)
print("Accuracy of the model using DT: " , acc_decision_tree)

# Running Random Forest model
from sklearn.ensemble import RandomForestClassifier
random_forest = RandomForestClassifier(n_estimators=100)
random_forest.fit(X_train, Y_train)
Y_pred = random_forest.predict(X_test)
random_forest.score(X_train, Y_train)
acc_random_forest = round(random_forest.score(X_train, Y_train) * 100, 2)
print("Accuracy of the model using RF: " , acc_random_forest)


models = pd.DataFrame({
    'Model': ['Logistic Regression','Support Vector Machines',  
              'Decision Tree','Random Forest'],
    'Score': [acc_log, acc_svc, acc_decision_tree, acc_random_forest]})
models.sort_values(by='Score', ascending=False)

print (models.plot(kind="bar",alpha=0.8))
