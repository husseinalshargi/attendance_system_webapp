import os
import streamlit as st



#important information: in order to see the page in a browser type in the terminal:
#ensure that you are using the venv
#1- cd (to the folder of the project)
#2- type: streamlit run (name of the file ex: Home.py)
# Rather than that run (streamlit run Home.py --server.fileWatcherType none) to solve the conflict problem of torch and streamlit

#here the codes start:
#set page config must be the first st code in the file

st.header('Attendence System using Face Recognition')


#show a spinner while downloading the data:
with st.spinner("Loading Models and Connecting to Redis db... "):
    import face_rec

st.success("Model loaded successfully")
st.success("Redis db successfully loaded")

