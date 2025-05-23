import numpy as np
import pandas as pd
import cv2

from dotenv import load_dotenv
import os
import redis

from insightface.app import FaceAnalysis
from sklearn.metrics import pairwise

import time
from datetime import datetime

#connect to redis db
load_dotenv()

host = os.getenv('REDIS_HOST')
port = int(os.getenv('REDIS_PORT'))
password = os.getenv('REDIS_PASSWORD')

r = redis.Redis(
    host=host,
    port=port,
    password=password
)

# retrive df from db
def retrive_features_df(name):
    retrive_dict = r.hgetall(name)
    retrive_series = pd.Series(retrive_dict)
    retrive_series = retrive_series.apply(lambda x: np.frombuffer(x, dtype=np.float32))
    index = retrive_series.index
    index = list(map(lambda x: x.decode(), index))
    retrive_series.index = index
    retrive_df = retrive_series.to_frame().reset_index()
    retrive_df.columns = ['name_role', 'facial_features']
    retrive_df[['Name','Role']] =retrive_df['name_role'].apply(lambda x: x.split('@')).apply(pd.Series)
    return retrive_df[['Name', 'Role', 'facial_features']]





# configure face analysis model
FaceApp = FaceAnalysis('buffalo_sc', 
                       'insightFace_models/buffalo_sc', 
                       providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])

FaceApp.prepare(0, det_size= (640, 640), 
                det_thresh=0.5)

# search for features

def ml_search(dataframe, feature_column, test_vector, name_role= ['Name', 'Role'], threshold=0.5):
    """
    using cosine similarity
    """
    #step 1 - take the dataframe
    dataframe = dataframe.copy()

    #step 2 - index face features (embedings) from the dataframe and convert into array
    x_list = dataframe[feature_column].tolist()
    x_array = np.asarray(x_list)

    #step 3 - Col. cosine similarity
    similarity = pairwise.cosine_similarity(x_array, test_vector.reshape(1, -1))
    similarity_arr = np.array(similarity).flatten()
    dataframe['cosine'] = similarity_arr

    #step 4 - filter data
    data_filtered = dataframe[dataframe['cosine'] >= threshold]

    #step 5 - get persons name
    if len(data_filtered) > 0:
        data_filtered.reset_index(drop=True, inplace=True)
        argmax = data_filtered['cosine'].argmax()
        person_name, person_role = data_filtered.loc[argmax][name_role]
    else:
        person_name, person_role = 'Unknown', 'Unknown'

    return person_name, person_role


#real-time prediction
# save logs for every min

class RealTimePred:
    def __init__(self):
        self.logs = dict(name=[], role=[], current_time=[])
    
    def reset_dict(self):
        self.logs = dict(name=[], role=[], current_time=[])

    def save_logs_to_db(self):
        #create a logs df
        df = pd.DataFrame(self.logs)
        #drop dublicated info (distinct name)
        df.drop_duplicates('name', inplace= True)
        #push data to db (list)
        #encode data
        name_list = df['name'].tolist()
        role_list = df['role'].tolist()
        ctime_list = df['current_time'].tolist()
        encoded_data = []
        for name, role, ctime in zip (name_list, role_list, ctime_list):
            if name != 'Unknown':
                concat_str = f'{name}@{role}@{ctime}'
                encoded_data.append(concat_str)
        
        if len(encoded_data) > 0:
            r.lpush('attendence:logs', *encoded_data)

        self.reset_dict()

    def face_prediction(self, test_image, dataframe, feature_column, name_role= ['Name', 'Role'], threshold=0.5):
        #step 0 - find the current time
        current_time = str(datetime.now())


        #step 1 - apply the test image to insight face
        results = FaceApp.get(test_image)
        test_copy = test_image.copy()

        #step 2 - for loop to extract all embeddings and pass them to ml_search
        for person in results:
            x1, y1, x2, y2 = person['bbox'].astype(int)
            embeddings = person['embedding']
            person_name, person_role = ml_search(dataframe, feature_column, embeddings, name_role=name_role, threshold= threshold)
            if person_name == 'Unknown' or person_role == 'Unknown': 
                color = (0,0,255)
            else:
                color = (0,255,0)
            cv2.rectangle(test_copy, (x1, y1), (x2, y2), color, 2)
            cv2.putText(test_copy, f'{person_name}, {person_role}', (x1 - 10, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            cv2.putText(test_copy, current_time, (x1, y2+10), cv2.FONT_HERSHEY_SIMPLEX, 0.3, color, 1)
            #save info in log dict
            self.logs['name'].append(person_name)
            self.logs['role'].append(person_role)
            self.logs['current_time'].append(current_time)


        return test_copy


# registration form:
class RegistrationForm:
    def __init__(self):
        self.sample = 0
    def reset(self):
        self.sample = 0    

    def get_embedding(self, frame):
        #get result from model
        result = FaceApp.get(frame, max_num=1)
        embeddings = None
        if not result:  # No face detected  ------------------------------
            return frame, None
        for res in result:
            self.sample +=1
            x1, y1, x2, y2 = res['bbox'].astype(int)
            cv2.rectangle(frame, (x1,y1), (x2,y2), (0,255,0), 1)

            #put text samples info
            text = f"samples = {self.sample}"
            cv2.putText(frame, text, (x1,y1), cv2.FONT_HERSHEY_DUPLEX, 0.6, (255,255,0), 2)

            #features
            embeddings = res['embedding']

            return frame, embeddings
        
    def save_data_in_redis_db(self, name, role):
        #validations:
        if name is not None:
            if name.strip() != '':
                key = f'{name}@{role}'
            else:
                return 'name_false'
        else:
            return 'name_false'

        # if the textfile exists
        if 'face_embedding.txt' not in os.listdir():
            return 'file_false'

        
        #step 1 - load face_embedding.txt
        x_array = np.loadtxt('face_embedding.txt', dtype= np.float32) #flatten array

        #step 2 - convert to array (a proper shape)
        received_samples = int(x_array.size/512)
        x_array = x_array.reshape(received_samples, 512)
        x_array = np.asarray(x_array)

        #step 3 - get mean
        x_mean = x_array.mean(axis= 0)
        x_mean_bytes = x_mean.tobytes()

        #step 4 - save into redis db
        # redis hashes
        r.hset(name= 'academy:register', key= key, value= x_mean_bytes)

        os.remove('face_embedding.txt')

        self.reset()

        return True