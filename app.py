import pickle5 as pickle
import numpy as np
import streamlit as st



# import the model
pipe = pickle.load(open('pipe.pkl','rb'))
laptop_data = pickle.load(open('laptop_data.pkl','rb'))

st.title('Laptop Predictor')

# brand
company = st.selectbox('Brand',laptop_data['Company'].unique())

# type of laptop
type = st.selectbox('Brand',laptop_data['TypeName'].unique())

# Ram
ram = st.selectbox('RAM(in GB',[2,4,6,8,12,16,24,32,64])

# weight
weight = st.number_input('Weight of the laptop')

# touchscreen
touchscreen = st.selectbox('Touchscreen',['No','Yes'])

# IPS
ips = st.selectbox('IPS',['No','Yes'])

# screen size
screen_size = st.number_input('Screen Size')

# resolution
resolution = st.selectbox('Screen Resolution',['1920x1080','1366x768','1600x900','3840x2160','3200x1800','2880x1800','2560x1440','2304x1440'])

# cpu
cpu = st.selectbox('Cpu',laptop_data['Cpu Brand'].unique())

# hard drive
hdd = st.selectbox('HDD(in GB)',[0,128,256,512,1024,2048])

ssd = st.selectbox('SSD(in GB)',[0,128,256,512,1024])

gpu = st.selectbox('GPU',laptop_data['Gpu Brand'].unique())

os = st.selectbox('OS',laptop_data['os'].unique())

if st.button('Predict Price'):

    ppi = None
    if touchscreen == 'Yes':
        touchscreen = 1
    else:
        touchscreen = 0

    if ips == 'Yes':
        ips =1
    else:
        ips =0

    x_res = int( resolution.split('x')[0])
    y_res = int(resolution.split('x')[1])
    ppi = ((x_res**2) + (y_res**2))**0.5/screen_size
    query = np.array([company,type,ram,weight,touchscreen,ips,ppi,cpu,hdd,ssd,gpu,os])
    query = np.array(query, dtype=object)

    query = query.reshape(1,12)
    st.title("Predicted price is Rs. " + str(int(np.exp(pipe.predict(query)))))
