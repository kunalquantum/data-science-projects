import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

df = pd.read_csv('heart.csv')

selected_features = ['oldpeak', 'thalachh', 'caa', 'cp', 'thall', 'age']
X = df[selected_features]
y = df['output']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

st.title('Heart Attack Readmission Risk Prediction')
st.image("heart.gif")

st.sidebar.header('Patient Information')

user_input = {}

for feature in selected_features:
    user_input[feature] = st.sidebar.slider(f'Select {feature}', float(X[feature].min()), float(X[feature].max()), float(X[feature].median()))

if st.sidebar.button('Predict Readmission Risk'):
    user_input_df = pd.DataFrame([user_input])
    prediction_prob = model.predict_proba(user_input_df)[0, 1]

    st.subheader('Prediction Result:')
    st.write(f'The predicted readmission risk probability: {prediction_prob:.2f}')

    if prediction_prob >= 0.5:
        st.image('highrisk.gif', caption='Readmission Predicted', use_column_width=True)
        st.title("--------------------High Risk--------------")
    else:
        st.image('lowrisk.gif', caption='No Readmission Predicted', use_column_width=True)
        st.title("--------------------Low Risk---------------")
