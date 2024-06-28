import joblib
import numpy as np
import streamlit as st

# Load the model and encoders
model = joblib.load(r'D:\titanic_flask_app\model\model.pkl')
le_sex = joblib.load(r'D:\titanic_flask_app\model\le_sex.pkl')
le_embarked = joblib.load(r'D:\titanic_flask_app\model\le_embarked.pkl')
scaler = joblib.load(r'D:\titanic_flask_app\model\scaler.pkl')

def predict_survival(pclass, sex, age, family_size, fare, embarked):
    sex_encoded = le_sex.transform([sex.lower()])[0]
    embarked_encoded = le_embarked.transform([embarked.upper()])[0]

    features = np.array([pclass, sex_encoded, age, family_size, fare, embarked_encoded]).reshape(1, -1)
    final_features = scaler.transform(features)
    prediction = model.predict(final_features)
    return prediction[0]

def main():
    st.title('Titanic Survival Prediction')

    # Input fields
    pclass = st.number_input('Class (1, 2, or 3)', min_value=1, max_value=3, step=1)
    sex = st.selectbox('Sex', ['male', 'female'])
    age = st.number_input('Age', min_value=0, max_value=100, step=1)
    family_size = st.number_input('Family Size', min_value=0, max_value=10, step=1)
    fare = st.number_input('Fare', min_value=0.0, max_value=1000.0, step=1.0)
    embarked = st.selectbox('Embarked', ['C', 'Q', 'S'])

    if st.button('Predict'):
        prediction = predict_survival(pclass, sex, age, family_size, fare, embarked)
        if prediction==1:
            st.write(f'Survival Prediction: Survive')
        else:
            st.write(f'Survival Prediction: Not Servive')
        

if __name__ == '__main__':
    main()
