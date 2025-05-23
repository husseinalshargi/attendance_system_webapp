from Home import st, face_rec
import time
from streamlit_webrtc import webrtc_streamer
import av

st.subheader('Real-Time Attendance System') #like <h2><h2/>



# Retrive the data from Redis db
with st.spinner("Retriving Data from Redis db..."):
    retrived_df = face_rec.retrive_features_df(name= 'academy:register')
    st.dataframe(retrived_df) #to show it in the app
st.success("Data successfully retrived from Redis")

# time
waitTime = 60 #time in sec
setTime =time.time()

realTimePred = face_rec.RealTimePred() #real time pred class




# Real-time Prediction
# in order to show the real-time window to the user we should use streamlit-webrtc
# callback function to show the video in real-time

def video_frame_callback(frame):
    global setTime
    img = frame.to_ndarray(format= 'bgr24') # 3 dim numpy array
    #here you can do anything to the image (array) before you return it
    pred_img = realTimePred.face_prediction(img, retrived_df,
                                         'facial_features', ['Name', "Role"], 0.5)

    timenow = time.time()
    diffTime = timenow - setTime
    if diffTime >= waitTime:
        realTimePred.save_logs_to_db()
        setTime = time.time()

        print('Saved Data to Redis db')

    return av.VideoFrame.from_ndarray(pred_img, format= 'bgr24')

webrtc_streamer(key= 'realtimePrediction', video_frame_callback=video_frame_callback,
    rtc_configuration={
        "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
    }
)   
                
