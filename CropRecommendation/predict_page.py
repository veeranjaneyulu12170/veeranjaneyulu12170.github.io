"""
import streamlit as st 
import pickle
import numpy as np

def load_model():
    with open('saved_steps.pkl', 'rb') as file:
     data=pickle.load(file)
    return data

data=load_model()

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

def show_predict_page():
    st.title("Road accident analysis")

    st.write(" Need some Information ")

    Age= (

        "18-30",
        "31-50",
        "Over 51",   
        "Under 18 "


    )
    education=(
           
           "Junior high school",
           "Elementary school",
           "High school",
           "Above high school",
           "Writing & reading",
           "Illiterate"


    )
    Age=st.selectbox("Age_band_of_driver",Age)
    education=st.selectbox("Educational_level",education)


show_predict_page()

"""





import streamlit as st 
import pickle
import numpy as np

label_map={0: 'apple', 1: 'banana', 2: 'blackgram', 3: 'chickpea', 4: 'coconut', 5: 'coffee', 6: 'cotton', 7: 'grapes', 8: 'jute', 9: 'kidneybeans', 10: 'lentil', 11: 'maize', 12: 'mango', 13: 'mothbeans', 14: 'mungbean', 15: 'muskmelon', 16: 'orange', 17: 'papaya', 18: 'pigeonpeas', 19: 'pomegranate', 20: 'rice', 21: 'watermelon'}

def load_model():
    with open('saved_steps.pkl', 'rb') as file:
     data=pickle.load(file)
    return data

data=load_model()

logistic_regression=data['classifier']

def show_predict_page():
    st.title("Crop Recommendation")

    N= st.number_input("Nitrogen Content ", step=0.1)
    st.write("You entered:", N)

    K= st.number_input("Potassium Content ", step=0.1)
    st.write("You entered:", K)

    P= st.number_input("Phosporous Content ", step=0.1)
    st.write("You entered:", P)

    temp= st.number_input("Temperature ", step=0.1)
    st.write("You entered:", temp)

    
    H = st.slider("Humidity ", 0.0, 100.0, 0.1)
    st.write("You selected:", H)

    
    ph = st.slider("PH ", 0.0, 14.0, 0.1)
    st.write("You selected:", ph)

    r= st.number_input("Rainfall(mm) ", step=0.1)
    st.write("You entered:", r)


    ok=st.button("Click Here")
    if ok:
       x=np.array([[N,K,P,temp,H,ph,r]])
       val=logistic_regression.predict(x)
       st.subheader(f"Recommended Crop is {label_map[val[0]]}")

show_predict_page()