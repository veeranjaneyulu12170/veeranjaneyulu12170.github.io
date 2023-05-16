import pandas as pd
import numpy as np
df=pd.read_csv('Crop_recommendation (1).csv')
print(df.head())

# 2..............

z=np.array(df['label'])


# 3.............

print(df['label'].value_counts())


# 4.............

df['label']=df['label'].astype('category')
df['labeln'] = df['label'].cat.codes
from sklearn.preprocessing import OneHotEncoder
enc = OneHotEncoder()

label = pd.DataFrame(enc.fit_transform(
    df[['labeln']]).toarray())

label = label.apply(lambda x: x.argmax(), axis=1).values

df.drop(columns=['label'],inplace=True)

df['label']=label

df.drop(columns=['labeln'],inplace=True)

print(df['label'].unique())


# 5...............

print(df['label'].value_counts())

# 6...............

targets=enc.fit_transform(z.reshape(-1,1)).toarray()

label_map={i:label for i, label in enumerate(enc.get_feature_names_out())}

print(label_map)


# 7.............

x=df.iloc[:,0:7]
y=df.iloc[:,-1]


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=0)


from sklearn.linear_model import LogisticRegression
model=LogisticRegression()


from sklearn.preprocessing import StandardScaler
scale=StandardScaler() 

x_train=scale.fit_transform(x_train)
x_test=scale.fit_transform(x_test)

model.fit(x_train,y_train)

y_pred=model.predict(x_test)

from  sklearn.metrics import accuracy_score
print(accuracy_score(y_pred,y_test))

from sklearn.metrics import classification_report
print('//////////////////////////////////////////////////////')
print('//////////////////////////////////////////////////////')
print('//////////////////////////////////////////////////////')
print(classification_report(y_pred,y_test))

val=[[45,69,85,77,89,7,99.56]]


predv=model.predict(val)
print(predv)

print(val[0])

import pickle

data={"classifier":model}

with open('saved_steps.pkl', 'wb') as file:
  pickle.dump(data,file)

with open('saved_steps.pkl', 'rb') as file:
  data=pickle.load(file)

logistic_regression=data['classifier']


predvn=logistic_regression.predict(val)
print(predvn)

