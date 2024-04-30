import streamlit as st
import pandas as pd
import pickle

# Load model
@st.cache
def load_model(model_path):
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    return model

# Function to preprocess input data
def preprocess_input(data):
    # No preprocessing needed as the data structure matches the model's input
    return data

# Function to make predictions
def predict(model, data):
    # Preprocess input data
    processed_data = preprocess_input(data)
    # Make predictions
    predictions = model.predict(processed_data)
    return predictions

# Load best model
model_path = "best_model.pkl"
best_model = load_model(model_path)

# Streamlit app
def main():
    st.title("Churn Prediction App")

    # Sidebar
    st.sidebar.title("Input Data")

    # Example input fields (modify as needed)
    credit_score = st.sidebar.number_input("Credit Score", min_value=300, max_value=850, value=700)
    geography = st.sidebar.selectbox("Geography", ["France", "Spain", "Germany"])
    gender = st.sidebar.selectbox("Gender", ["Male", "Female"])
    age = st.sidebar.number_input("Age", min_value=18, max_value=100, value=30)
    tenure = st.sidebar.number_input("Tenure", min_value=0, max_value=20, value=5)
    balance = st.sidebar.number_input("Balance", min_value=0.0, value=0.0)
    num_of_products = st.sidebar.number_input("Number of Products", min_value=1, max_value=4, value=1)
    has_cr_card = st.sidebar.selectbox("Has Credit Card", ["No", "Yes"])
    is_active_member = st.sidebar.selectbox("Is Active Member", ["No", "Yes"])
    estimated_salary = st.sidebar.number_input("Estimated Salary", min_value=0.0, value=50000.0)

    # Map categorical variables to numerical values
    geography_mapping = {"France": 0, "Spain": 1, "Germany": 2}
    gender_mapping = {"Male": 0, "Female": 1}
    has_cr_card_mapping = {"No": 0, "Yes": 1}
    is_active_member_mapping = {"No": 0, "Yes": 1}

    geography = geography_mapping[geography]
    gender = gender_mapping[gender]
    has_cr_card = has_cr_card_mapping[has_cr_card]
    is_active_member = is_active_member_mapping[is_active_member]

    # Create DataFrame from input data
    input_data = pd.DataFrame({
        'CreditScore': [credit_score],
        'Geography': [geography],
        'Gender': [gender],
        'Age': [age],
        'Tenure': [tenure],
        'Balance': [balance],
        'NumOfProducts': [num_of_products],
        'HasCrCard': [has_cr_card],
        'IsActiveMember': [is_active_member],
        'EstimatedSalary': [estimated_salary]
    })

    # Make predictions
    if st.sidebar.button("Predict"):
        prediction = predict(best_model, input_data)
        churn_status = "Churn" if prediction[0] == 1 else "Not Churn"
        st.write("Prediction:", churn_status)

if __name__ == "__main__":
    main()
