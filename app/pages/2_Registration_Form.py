from Home import st, face_rec
from streamlit_webrtc import webrtc_streamer
import av

st.subheader('Registeration From') #like <h2><h2/>

#init reg form class
registrationForm = face_rec.RegistrationForm()


#step 1 - collect person's name and role
#form
person_name = st.text_input(label= 'Name', placeholder= 'First & Last Name')
role = st.selectbox('Select Your Role', options= ('Student', 'Teacher'))



#step 2 - collect facial features of that person
def video_callback(frame):
    img = frame.to_ndarray(format= 'bgr24')
    rec_img, embedding = registrationForm.get_embedding(img)
    #two step process
    #1- save data locally -> txt
    if embedding is not None:
        with open('face_embedding.txt', mode= 'ab') as f:
            face_rec.np.savetxt(f, embedding.astype(face_rec.np.float32))
    
    return av.VideoFrame.from_ndarray(rec_img, format= 'bgr24')


webrtc_streamer(key= 'registration', video_frame_callback=video_callback,
   rtc_configuration={
        "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
    }             
                )


#step 3 - save data in redis db


if st.button('submit'):
    return_val = registrationForm.save_data_in_redis_db(person_name, role)
    if return_val == True:
        st.success(f"{person_name} registered successfully.")
    elif return_val == 'name_false':
        st.error('Please enter the name: name connot be empty or spaces.')
    elif return_val == 'file_false':
        st.error('face_embedding.txt is not found, try again later.')
