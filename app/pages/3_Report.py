from Home import st, face_rec


st.subheader('Reporting') #like <h2><h2/>

#retrive logs data and show in the page
#extract data from redis db
name = 'attendence:logs'
def load_logs(name, end= -1):
    logs_list = face_rec.r.lrange(name, 0, -1) #extract all data from redis db
    return logs_list

#taps to show the info
tab1, tab2 = st.tabs(['Register Data', 'Logs'])

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

        with col2:
            if st.button("Cancel", key="cancel_delete"):
                st.info("Delete canceled.")
                del st.session_state.confirm_delete  # clear state after cancel

with tab2:
    if st.button('refresh Logs'):
        st.write(load_logs(name= name))

    





