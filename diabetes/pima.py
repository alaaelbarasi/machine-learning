from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import numpy as np 
import pandas as pd
df=pd.read_csv('diabetes.csv')
col=['Pregnancies','Glucose','BloodPressure','SkinThickness','Insulin','BMI','DiabetesPedigreeFunction','Age']
labels=df['Outcome'].values
features= df[list(col)].values
X_train , X_test, y_train, y_test = train_test_split(features, labels , test_size=0.30)
clf = RandomForestClassifier(n_estimators=12)
clf=clf.fit(X_train,y_train)
accuracy = clf.score(X_test,y_test)
print('Accuracy =' ,accuracy*100)