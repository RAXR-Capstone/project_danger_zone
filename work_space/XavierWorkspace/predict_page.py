import numpy as np
import requests
from requests.api import get
from env import token
import pandas as pd
import streamlit as st
from streamlit_folium import folium_static
import pickle
import folium
import time
from ipgetter2 import IPGetter, ipgetter1 as ipgetter
import ipinfo


def load_model():
    with open('model.pkl', 'rb') as file:
        data = pickle.load(file)
    return data

def load_data():
    with open('data.pkl', 'rb') as file:
        data = pickle.load(file)
    return data

model = load_model()
the_model = model['model']

data = load_data()
the_data = data['data']


def show_predict_page():
    st.title('Plot My Injury')
    st.write('''### Enter Demographic Infomation Here''')
    age = st.number_input(label='driver  age', step = 1)
    car_make = st.selectbox('vehicle make', sorted(the_data.vehicle_make.unique()))
    car_year = st.number_input(label='car year', step = 1)
    make_type = st.selectbox('make',sorted(the_data[the_data['vehicle_make']== car_make]['vehicle_type'].value_counts().index.tolist()))
    number_of_occupants = st.number_input(label='Number of Occupants', step = 1)
    button = st.checkbox('Run')
    if button:
        ip = requests.get('https://api.ipify.org').text

        access_token = token
        handler = ipinfo.getHandler(access_token)
        ip_address = ip
        details = handler.getDetails(ip_address)
        latlong = str(details.loc).split(',')


        # center on San Antonio
        m = folium.Map(location=[29.377711363953658, -98.4970935625], zoom_start=10)
        folium.CircleMarker(location=[latlong[0], latlong[1]], radius = 0.5, color='blue').add_to(m)

     

        # call to render Folium map in Streamlit
        folium_static(m)

