from Home import st, face_rec


st.subheader('Reporting') #like <h2><h2/>

#retrive logs data and show in the page
#extract data from redis db
name = 'attendence:logs'
def load_logs(name, end= -1):
    logs_list = face_rec.r.lrange(name, 0, end) #extract all data from redis db
    return logs_list

#taps to show the info
tab1, tab2, tab3 = st.tabs(['Register Data', 'Logs', "Attendance Report" ])

with tab1:
    if st.button('Refersh Data'):
        with st.spinner("Retriving Data from Redis db..."):
            retrived_df = face_rec.retrive_features_df(name= 'academy:register')
            st.dataframe(retrived_df[['Name', 'Role']]) #to show it in the app


    #to delete:
    records_name = 'academy:register'
    persons = face_rec.retrive_features_df(records_name)
    persons_names = persons['Name'].tolist()
    persons_roles = persons['Role'].tolist()
    name_to_delete = st.selectbox('Select a name to delete: ', options= persons_names)
    
    if st.button("Delete", key= 'delete_main'): 
        # Set a session state variable to show confirmation
        st.session_state.confirm_delete = name_to_delete

    if "confirm_delete" in st.session_state:
        st.warning(f"Are you sure you want to delete '{st.session_state.confirm_delete}'?")
        col1, col2 = st.columns(2)
    
        with col1:
            if st.button("Yes, Delete", key="yes_delete"):
                # Correctly find the name and role
                selected_name = st.session_state.confirm_delete
                role = persons_roles[persons_names.index(selected_name)]
                personToDelete = f'{selected_name}@{role}'
                # Delete from Redis
                face_rec.r.hdel(records_name, personToDelete)
                st.success(f"'{st.session_state.confirm_delete}' deleted.")
                del st.session_state.confirm_delete  # clear state after action
                st.rerun()

        with col2:
            if st.button("Cancel", key="cancel_delete"):
                st.info("Delete canceled.")
                del st.session_state.confirm_delete  # clear state after cancel
                st.rerun()

with tab2:
    if st.button('refresh Logs'):
        st.write(load_logs(name= name))

    
with tab3:
    st.subheader('Attendance Report')

    #load logs into a list
    logs_list = load_logs(name= name)
    
    #convert from list of bytes to list of strings
    convertToString = lambda x: x.decode('utf-8')
    logs_list_string = list(map(convertToString, logs_list))

    # split @ from strings
    split_string = lambda x: x.split('@')
    logs_nested_list = list(map(split_string, logs_list_string))

    #make the list to df
    logs_df = face_rec.pd.DataFrame(logs_nested_list, columns= ['Name', 'Role', 'Timestamp'])

    # time based analysis (report)
    logs_df['Timestamp'] = face_rec.pd.to_datetime(logs_df['Timestamp'], format='mixed', errors='coerce')


    logs_df['Date'] = logs_df['Timestamp'].dt.date

    # finding in - out times
    # in time: first time a pesron detected in a day (min time stamp of the day)
    # out time: last time a pesron detected in a day (max time stamp of the day)

    report_df = logs_df.groupby(by= ['Date', 'Name', 'Role']).agg(
        InTime = face_rec.pd.NamedAgg('Timestamp', 'min'), #in time
        OutTime = face_rec.pd.NamedAgg('Timestamp', 'max') #out time
    ).reset_index()


    report_df["Duration"] = report_df['InTime'] - report_df['OutTime']

    #make it only the time
    report_df['InTime'] = report_df['InTime'].dt.time
    report_df['OutTime'] = report_df['OutTime'].dt.time

    #makig a person present or absent
    all_dates = report_df['Date'].unique()
    name_role = report_df[['Name', 'Role']].drop_duplicates().values.tolist()

    date_name_role_zip = []
    for dt in all_dates:
        for name, role in name_role:
            date_name_role_zip.append([dt, name, role])
    
    date_name_role_zip_df = face_rec.pd.DataFrame(date_name_role_zip, columns=['Date', 'Name', 'Role'])

    # left join with report_df
    date_name_role_zip_df = face_rec.pd.merge(date_name_role_zip_df, report_df, how='left', on= ['Date', 'Name', 'Role'])



    #duaration
    date_name_role_zip_df['Duaration_seconds'] = date_name_role_zip_df['Duration'].dt.seconds
    date_name_role_zip_df['Duaration_hours'] = date_name_role_zip_df['Duaration_seconds'] / (60*60)


    def status_marker(x):
        if face_rec.pd.Series(x).isnull().all():
            return 'Absent'
        elif x>=0 and x<1:
            return 'Absent (less than 1 hr)'
        elif x>=1 and x<4:
            return 'Half day (less than 4 hours)'
        elif x>=4 and x<6:
            return 'Half Day'
        elif x>=6:
            return 'Present'
    
    date_name_role_zip_df['Status'] = date_name_role_zip_df['Duaration_hours'].apply(status_marker)

    #tab 
    t1, t2 = st.tabs(['Complete Report', 'Filter Report'])

    with t1: 
        st.subheader('Complete Report')
        st.dataframe(date_name_role_zip_df)

    with t2:
        st.subheader('Search Records')

        #Date filter
        date_in = str(st.date_input('Filter Date', face_rec.datetime.now().date()))

        #Names filter 
        name_list = date_name_role_zip_df['Name'].unique().tolist()
        name_in = st.selectbox('Select Name', ['ALL'] + name_list, key='name_in')

        #Teachers or Students filter
        role_list = date_name_role_zip_df['Name'].unique().tolist()
        role_in = st.selectbox('Select Name', ['ALL'] + role_list, key='role_in')

        # Filter by duration
        duration_in = st.slider('Filter the duration in hours grater than ', 0, 15, 6, key='duration_in')

        #status filter
        status_list = date_name_role_zip_df['Status'].unique().tolist()
        status_in = st.multiselect('Select Name', ['ALL'] + status_list, key='status_in') #return list

        if st.button('Submit', key='SubmitFilter'):
            date_name_role_zip_df['Date'] = date_name_role_zip_df['Date'].astype(str)
            
            #filter date
            filter_df = date_name_role_zip_df.query(f'Date == "{date_in}"')

            #filter name
            if name_in != 'ALL':
                filter_df = filter_df.query(f'Name == "{name_in}"')

            #filter role
            if role_in != 'ALL':
                filter_df = filter_df.query(f'Role == "{role_in}"')

            #filter duration
            if duration_in > 0:
                filter_df = filter_df.query(f'Duaration_hours > "{duration_in}"')
            
            #filter duration
            if 'ALL' in status_in:
                filter_df= filter_df

            elif len(status_in) > 0:
                filter_df['staatus_condition'] = filter_df['Status'].apply(lambda x: True if x in status_in else False)
                filter_df = filter_df.query(f'staatus_condition == True')
                filter_df.drop(columns= 'status_condition', inplace= True)
            
            else:
                filter_df = filter_df


            st.dataframe(filter_df)









