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
