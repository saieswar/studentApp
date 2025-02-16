import pickle
import pandas as pd
import streamlit as st

def load_model():
    with open('linear_stundent.pkl', 'rb') as file:
        model,scalar,le = pickle.load(file)
    return model, scalar, le


def pre_process_model(data, scalar, le):
    data['Extracurricular Activities']= le.transform([data['Extracurricular Activities']])[0]
    df = pd.DataFrame([data])
    transformed_df = scalar.transform(df)
    return transformed_df


def predict_model(data):
    model,scalar, le = load_model()
    preprossed_data = pre_process_model(data, scalar, le)
    prediction = model.predict(preprossed_data)
    return prediction

def main():
    st.title('Student Performance Analysis and Decision Model')
    st.write("Enter your Student Data to Get perforance prediction")

    hours_studied = st.number_input("Hours of Study", min_value = 1, max_value = 10 , value = 5)
    previous_score = st.number_input("previous Score", min_value = 1, max_value =100,value=50)
    extracurricular_activities = st.selectbox("Select Extracurricular Activities", ['Yes', 'No'])
    sleep_hours = st.number_input("Sleep Hours", min_value = 4, max_value = 14 , value = 7)
    sample_question_papers = st.number_input("Sample Question Papers", min_value = 1, max_value = 20 , value = 5)

    if st.button('Predict your Score'):
        user_data = {
        'Hours Studied': hours_studied,
        'Previous Scores': previous_score,
        'Extracurricular Activities': extracurricular_activities,
        'Sleep Hours': sleep_hours,
        'Sample Question Papers Practiced': sample_question_papers
        }
        
        prediction  = predict_model(user_data)
        
        st.success(f"Based on your input, your predicted performance score is: {prediction}")

if __name__ == '__main__':
    main()
