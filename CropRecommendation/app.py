"""
import pandas as pd
import numpy as np
df=pd.read_csv("RTA Dataset.csv")
print(df.head(5))

df=df[['Age_band_of_driver','Educational_level','Vehicle_driver_relation','Driving_experience','Lanes_or_Medians','Types_of_Junction','Road_surface_type','Light_conditions','Weather_conditions','Type_of_collision','Vehicle_movement','Pedestrian_movement','Cause_of_accident','Accident_severity']]
print(df.head(5))

df.replace(['unknown','Unknown','nan'],np.NaN,inplace=True)
print(df.isnull().sum())

df=df.dropna()
print(df.isnull().sum())

from sklearn.preprocessing import LabelEncoder
le_age=LabelEncoder()
df['Age_band_of_driver']=le_age.fit_transform(df['Age_band_of_driver'])
print(df['Age_band_of_driver'].unique())

le_education=LabelEncoder()
df['Educational_level']=le_education.fit_transform(df['Educational_level'])
df['Educational_level'].unique()

le_severity=LabelEncoder()
df['Accident_severity']=le_severity.fit_transform(df['Accident_severity'])
df['Accident_severity'].unique()

le_junctions=LabelEncoder()
df['Types_of_Junction']=le_junctions.fit_transform(df['Types_of_Junction'])
df['Types_of_Junction'].unique()

le_lanes=LabelEncoder()
df['Lanes_or_Medians']=le_lanes.fit_transform(df['Lanes_or_Medians'])
df['Lanes_or_Medians'].unique()

le_experience=LabelEncoder()
df['Driving_experience']=le_experience.fit_transform(df['Driving_experience'])
df['Driving_experience'].unique()

le_relation=LabelEncoder()
df['Vehicle_driver_relation']=le_relation.fit_transform(df['Vehicle_driver_relation'])
df['Vehicle_driver_relation'].unique()

le_collosion=LabelEncoder()
df['Type_of_collision']=le_collosion.fit_transform(df['Type_of_collision'])
df['Type_of_collision'].unique()

le_weather=LabelEncoder()
df['Weather_conditions']=le_weather.fit_transform(df['Weather_conditions'])
df['Weather_conditions'].unique()

le_light=LabelEncoder()
df['Light_conditions']=le_light.fit_transform(df['Light_conditions'])
df['Light_conditions'].unique()

le_roadtype=LabelEncoder()
df['Road_surface_type']=le_roadtype.fit_transform(df['Road_surface_type'])
df['Road_surface_type'].unique()

le_cause=LabelEncoder()
df['Cause_of_accident']=le_cause.fit_transform(df['Cause_of_accident'])
df['Cause_of_accident'].unique()

le_pedestrian=LabelEncoder()
df['Pedestrian_movement']=le_pedestrian.fit_transform(df['Pedestrian_movement'])
df['Pedestrian_movement'].unique()

le_movement=LabelEncoder()
df['Vehicle_movement']=le_movement.fit_transform(df['Vehicle_movement'])
df['Vehicle_movement'].unique()

print(df.dtypes)

print(df.head(5))

x=df.drop('Accident_severity',axis=1)
y=df['Accident_severity']

from sklearn.ensemble import RandomForestClassifier
classifier=RandomForestClassifier(random_state=0)
classifier.fit(x,y.values)

y_predicted=classifier.predict(x)

from sklearn.metrics import mean_squared_error,mean_absolute_error
error=np.sqrt(mean_squared_error(y,y_predicted))

print(error)

x=np.array([['18-30','Junior high school','Employee','5-10yr','Undivided Two way','Y Shape','Asphalt roads','Darkness - lights lit','Normal','Collision with roadside objects','Going straight','Not a Pedestrian','Overtaking']])

x[:,0]=le_age.transform(x[:,0])
x[:,1]=le_education.transform(x[:,1])
x[:,2]=le_relation.transform(x[:,2])
x[:,3]=le_experience.transform(x[:,3])
x[:,4]=le_lanes.transform(x[:,4])
x[:,5]=le_junctions.transform(x[:,5])
x[:,6]=le_roadtype.transform(x[:,6])
x[:,7]=le_light.transform(x[:,7])
x[:,8]=le_weather.transform(x[:,8])
x[:,9]=le_collosion.transform(x[:,9])
x[:,10]=le_movement.transform(x[:,10])
x[:,11]=le_pedestrian.transform(x[:,11])
x[:,12]=le_cause.transform(x[:,12])

x=x.astype(float)
print(x)

y_pred=classifier.predict(x)

print(y_pred)

import pickle

data={"model":classifier, "le_age":le_age,"le_education":le_education,"le_relation":le_relation,"le_experience":le_experience,"le_lanes":le_lanes,"le_junctions":le_junctions,"le_roadtype":le_roadtype,"le_light":le_light,"le_weather":le_weather,"le_collosion":le_collosion,"le_movement":le_movement,"le_pedestrian":le_pedestrian,"le_cause":le_cause}
with open('saved_steps.pkl', 'wb') as file:
  pickle.dump(data,file)

with open('saved_steps.pkl', 'rb') as file:
  data=pickle.load(file)

random_forest=data['model']
le_age=data['le_age']
le_education=data['le_education']
le_relation=data['le_relation']
le_experience=data['le_experience']
le_lanes=data['le_lanes']
le_junctions=data['le_junctions']
le_roadtype=data['le_roadtype']
le_light=data['le_light']
le_weather=data['le_weather']
le_collosion=data['le_collosion']
le_movement=data['le_movement']
le_pedestrian=data['le_pedestrian']
le_cause=data['le_cause']

y_pred=random_forest.predict(x)
print(y_pred)


"""

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

