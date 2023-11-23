from sklearn.neighbors import KNeighborsClassifier
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go

st.header('kairung')
#st.image("./pic/kairung.jpg")
col1, col2, col3 = st.columns(3)

with col1:
   st.header("Versicolor")
   st.image("./pic/iris1.jpg")

with col2:
   st.header("Verginiga")
   st.image("./pic/iris2.jpg")

with col3:
   st.header("Setosa")
   st.image("./pic/iris3.jpg")

html_7 = """
<div style="background-color:#EC7063;padding:15px;border-radius:15px 15px 15px 15px;border-style:'solid';border-color:black">
<center><h5>สถิติข้อมูลดอกไม้</h5></center>
</div>
"""
st.markdown(html_7, unsafe_allow_html=True)
st.markdown("")

dt = pd.read_csv("./data/Root1.csv")
st.write(dt.head(10))
html_8 = """
<div style="background-color:#6BD5DA;padding:15px;border-radius:15px 15px 15px 15px;border-style:'solid';border-color:black">
<center><h5>ทำนายข้อมูล</h5></center>
</div>
"""
st.markdown(html_8, unsafe_allow_html=True)
st.markdown("")

pt_len = st.slider("กรุณาเลือกข้อมูล petal.length")
pt_wd = st.slider("กรุณาเลือกข้อมูล petal.width")

sp_len = st.number_input("กรุณาเลือกข้อมูล sepal.length")
sp_wd = st.number_input("กรุณาเลือกข้อมูล sepal.width")


if st.button("ทำนายผล"):
    #st.write("ทำนาย")
   dt = pd.read_csv("./data/Root1.csv") 
   dt['Dependents'] = pd.to_numeric(dt['Dependents'])
   st.write(dt.dtypes)  
   X = dt.drop('Credit_History', axis=1)   
   y = dt.Credit_History 

   

   Knn_model = KNeighborsClassifier(n_neighbors=3)
   Knn_model.fit(X, y)  

   x_input = np.array([[0, 1, 1.4, 0,1,57.7,38,10,18.9]])
   st.write(Knn_model.predict(x_input))
   
   out=Knn_model.predict(x_input)

   if out[0] == '1':
    st.write('1')
   else:
    st.write('0')
else:
    st.write("ไม่ทำนาย")

