import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import pickle
import streamlit as st

# Load the dataset
data = pd.read_csv("domestic.csv")

# Select relevant columns and drop missing values
df_selected = data[["city1", "city2", "passengers", "fare_lg", "quarter", "carrier_lg"]]
df_selected.dropna(inplace=True)

# Initialize label encoders for the categorical columns
le_city1 = LabelEncoder()
le_city2 = LabelEncoder()
le_quarter = LabelEncoder()
le_carrier = LabelEncoder()

# Encode the categorical columns
df_selected["city1_encoded"] = le_city1.fit_transform(df_selected["city1"])
df_selected["city2_encoded"] = le_city2.fit_transform(df_selected["city2"])
df_selected["quarter_encoded"] = le_quarter.fit_transform(df_selected["quarter"])
df_selected["carrier_encoded"] = le_carrier.fit_transform(df_selected["carrier_lg"])

# Define features and target
X = df_selected[["city1_encoded", "city2_encoded", "quarter_encoded", "carrier_encoded"]]
y = df_selected["fare_lg"]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict and evaluate the model
y_pred = model.predict(X_test)
# print(f"Mean Squared Error: {mean_squared_error(y_test, y_pred)}")
# print(f"Mean Absolute Error: {mean_absolute_error(y_test, y_pred)}")
# print(f"R2 Score: {r2_score(y_test, y_pred)}")

# Save the trained model and encoders
with open("simple_model.pkl", "wb") as f:
    pickle.dump(model, f)

with open("encoders.pkl", "wb") as f:
    pickle.dump((le_city1, le_city2, le_quarter, le_carrier), f)

# Streamlit app
st.title("Flight Price Predictor")

# Load the model and encoders
with open("simple_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("encoders.pkl", "rb") as f:
    le_city1, le_city2, le_quarter, le_carrier = pickle.load(f)

# Get unique values for the dropdowns
city1_options = le_city1.classes_.tolist()
city2_options = le_city2.classes_.tolist()
quarter_options = le_quarter.classes_.tolist()
carrier_options = le_carrier.classes_.tolist()

quarter_display_map = {
    1: 'January - March',
    2: 'April - June',
    3: 'July - September',
    4: 'October - December'
}

pretty_names = {
'9N': 'Tropic Air',
    'A7': 'Unknown',  # A7 isn't a standard IATA airline code
    'AA': 'American Airlines',
    'AS': 'Alaska Airlines',
    'B6': 'JetBlue Airways',
    'CO': 'Continental Airlines',
    'DH': 'Independence Air',
    'DL': 'Delta Air Lines',
    'F9': 'Frontier Airlines',
    'FL': 'AirTran Airways',
    'G4': 'Allegiant Air',
    'HP': 'America West Airlines',
    'J7': 'ValuJet Airlines',
    'JI': 'Midway Airlines',
    'KP': 'Kiwi International Air Lines',
    'KW': 'Carnival Air Lines',
    'MX': 'Mexicana',
    'N7': 'National Airlines',
    'NJ': 'VivaAerobus',
    'NK': 'Spirit Airlines',
    'NW': 'Northwest Airlines',
    'PN': 'Pan Am',
    'QQ': 'Reno Air',
    'QX': 'Horizon Air',
    'RP': 'Chautauqua Airlines',
    'RU': 'Unknown',
    'SX': 'Skybus Airlines',
    'SY': 'Sun Country Airlines',
    'T3': 'Eastern Air Lines',
    'TW': 'TWA',
    'TZ': 'ATA Airlines',
    'U5': 'USA3000 Airlines',
    'UA': 'United Airlines',
    'US': 'US Airways',
    'VX': 'Virgin America',
    'W7': 'Western Pacific Airlines',
    'W9': 'Wizz Air',
    'WN': 'Southwest Airlines',
    'WV': 'Air Wisconsin',
    'XJ': 'Mesaba Airlines',
    'XP': 'Casino Express',
    'YV': 'Mesa Airlines',
    'YX': 'Republic Airlines',
    'ZA': 'AccessAir',
    'ZW': 'Air Wisconsin'
}

reverse_names = {v: k for k, v in pretty_names.items()}
carrier_options_pretty = [pretty_names.get(c, c) for c in le_carrier.classes_]
quarter_options = [quarter_display_map[q] for q in le_quarter.classes_]
display_to_original_quarter = {v: k for k, v in quarter_display_map.items()}


# Streamlit form for input
with st.form("input_form"):
    st.subheader("Enter Route Information")
    
    city1 = st.selectbox("Origin City", city1_options)
    city2 = st.selectbox("Destination City", city2_options)
    quarter = st.selectbox("Quarter", quarter_options)  # Human-readable
    carrier = st.selectbox("Carrier", carrier_options_pretty)  # Use carrier_options_pretty here

    submitted = st.form_submit_button("Predict")

    if submitted:
        city1_encoded = le_city1.transform([city1])[0]
        city2_encoded = le_city2.transform([city2])[0]
        quarter_original = display_to_original_quarter[quarter]
        quarter_encoded = le_quarter.transform([quarter_original])[0]
        
        # Find the corresponding carrier code from the pretty_names dictionary
        carrier_code = reverse_names.get(carrier, carrier)  # Get the original carrier code
        carrier_encoded = le_carrier.transform([carrier_code])[0]  # Using the original carrier code
        
        input_data = np.array([[city1_encoded, city2_encoded, quarter_encoded, carrier_encoded]])
        predicted_fare = model.predict(input_data)[0]
        st.success(f"Estimated Fare: ${predicted_fare:.2f}")