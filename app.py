
import streamlit as st
import joblib
import pandas as pd

# Load the trained model
model = joblib.load('logi.sav')

st.title('Delivery Delay Prediction App')
st.write('Enter the feature values below to predict the delivery delay (0 = No Delay, 1 = Delay).')

# Define input fields for each feature based on x.columns
delivery_distance = st.number_input('Delivery Distance', min_value=0.0, value=19.35)
traffic_congestion = st.selectbox('Traffic Congestion (1-5, 5 being high)', options=[1, 2, 3, 4, 5], index=3)
weather_condition = st.selectbox('Weather Condition (1-5, 5 being severe)', options=[1, 2, 3, 4, 5], index=2)
delivery_slot = st.selectbox('Delivery Slot (1-3)', options=[1, 2, 3], index=1)
driver_experience = st.number_input('Driver Experience (years)', min_value=0, value=16)
num_stops = st.number_input('Number of Stops', min_value=0, value=6)
vehicle_age = st.number_input('Vehicle Age (years)', min_value=0, value=9)
road_condition_score = st.selectbox('Road Condition Score (1-5, 5 being excellent)', options=[1, 2, 3, 4, 5], index=2)
package_weight = st.number_input('Package Weight', min_value=0.0, value=33.62)
fuel_efficiency = st.number_input('Fuel Efficiency', min_value=0.0, value=13.01)
warehouse_processing_time = st.number_input('Warehouse Processing Time (minutes)', min_value=0, value=58)

# Create a DataFrame from the inputs
input_data = pd.DataFrame([{
    'Delivery_Distance': delivery_distance,
    'Traffic_Congestion': traffic_congestion,
    'Weather_Condition': weather_condition,
    'Delivery_Slot': delivery_slot,
    'Driver_Experience': driver_experience,
    'Num_Stops': num_stops,
    'Vehicle_Age': vehicle_age,
    'Road_Condition_Score': road_condition_score,
    'Package_Weight': package_weight,
    'Fuel_Efficiency': fuel_efficiency,
    'Warehouse_Processing_Time': warehouse_processing_time
}])

# Make prediction when button is clicked
if st.button('Predict Delivery Delay'):
    prediction = model.predict(input_data)[0]
    prediction_proba = model.predict_proba(input_data)[0]

    st.subheader('Prediction Result:')
    if prediction == 1:
        st.error(f'Predicted Delivery Delay: Yes (Probability: {prediction_proba[1]:.2f})')
    else:
        st.success(f'Predicted Delivery Delay: No (Probability: {prediction_proba[0]:.2f})')
