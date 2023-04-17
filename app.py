from flask import Flask, request, render_template
import joblib
import numpy as np

import pandas as pd
# Load the trained model
model = joblib.load('taxi_fare_model.pkl')

# Create the Flask app
app = Flask(__name__)

# Define the home page
@app.route('/')
def home():
    return render_template('home.html')

# Define the prediction page
@app.route('/predict', methods=['POST'])
def predict():
    # Extract input features from the form
    distance = float(request.form['distance'])
    time = int(request.form['time'])
    passengers = int(request.form['passengers'])
    payment_type = request.form['payment_type']
    weather_condition = request.form['weather_condition']
    traffic_condition = request.form['traffic_condition']

    # Create a new DataFrame with the same column names as the training data
      # One-hot encode categorical variables
    df = pd.DataFrame({'distance': [distance], 'time': [time], 'passengers': [passengers], 'payment_type': [payment_type], 'weather_condition': [weather_condition], 'traffic_condition': [traffic_condition]})
    # df = pd.get_dummies(df, columns=['weather_condition','traffic_condition','payment_type'])
    
    df['speed'] = df['distance'] / (df['time'] / 60)
    df['fare_amount'] = 0
    df['payment_type_credit card'] = 0
    df = df.assign(weather_condition_clear=np.where(df['weather_condition'] == 'clear', 1, 0),
               weather_condition_cloudy=np.where(df['weather_condition'] == 'cloudy', 1, 0),
               weather_condition_rainy=np.where(df['weather_condition'] == 'clear', 1, 0),
               
               traffic_condition_heavy=np.where(df['traffic_condition'] == 'heavy', 1, 0),
               traffic_condition_light=np.where(df['traffic_condition'] == 'light', 1, 0),
               traffic_condition_moderate=np.where(df['traffic_condition'] == 'moderate', 1, 0),
               
               
               
               payment_type_cash=np.where(df['payment_type'] == 'cash', 1, 0),
               payment_type_credit_card=np.where(df['payment_type'].str.contains('credit'), 1, 0)
               )
    df = df[['distance', 'time', 'passengers',
       'weather_condition_clear', 'weather_condition_cloudy',
       'weather_condition_rainy', 'traffic_condition_heavy',
       'traffic_condition_light', 'traffic_condition_moderate',
       'payment_type_cash', 'payment_type_credit card', 'speed']]
    print(df.size)
    print(df.columns)
    # Select only the relevant features for the model
    # relevant_features = df.reindex(columns=df.columns, fill_value=0)

    # Predict the fare amount using the trained model
    fare_pred = model.predict(df)

    # Render the predicted fare amount on the web page
    return render_template('result.html', fare=fare_pred)




if __name__ == '__main__':
    # Run the Flask app
    app.run(debug=True)
