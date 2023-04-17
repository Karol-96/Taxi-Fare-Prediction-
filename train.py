import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

# Load the data set from the CSV file
df = pd.read_csv('taxi_fare_data_with_location.csv')
print(df)

df = df[['distance','time','passengers','fare_amount','weather_condition','traffic_condition','payment_type']]
# Check for missing values
print(df.isnull().sum())

# Check for duplicates
print(df.duplicated().sum())

# Drop duplicates
df.drop_duplicates(inplace=True)

# Remove outliers
Q1 = df['fare_amount'].quantile(0.25)
Q3 = df['fare_amount'].quantile(0.75)
IQR = Q3 - Q1
data = df[(df['fare_amount'] >= Q1 - 1.5*IQR) & (df['fare_amount'] <= Q3 + 1.5*IQR)]



# One-hot encoding categorical variables
df = pd.get_dummies(df, columns=['weather_condition','traffic_condition','payment_type'])


df = df.rename(columns=lambda x: x.replace(' ', '_'))

# Import necessary libraries
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import numpy as np

# Create new feature for average speed
df['speed'] = df['distance'] / (df['time'] / 60)

# df = df[['distance','time','passengers','temperature','fare_amount']]

# Split the data into train and test sets
X = df.drop(['fare_amount'], axis=1)
y = df['fare_amount']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(X_train.columns)
print(X_train.dtypes)

# Train the Ridge regression model with L2 regularization
model = Ridge(alpha=0.5)
model.fit(X_train, y_train)

# Evaluate the model on the test set
y_pred = model.predict(X_test)
r2 = r2_score(y_test, y_pred)
print('R-squared score:', r2)

# Randomly select a trip and predict the fare amount
trip = df.sample(1, random_state=42)
trip['speed'] = trip['distance'] / (trip['time'] / 60)
fare_pred = model.predict(trip.drop(['fare_amount'], axis=1))

print('Actual fare amount:', trip['fare_amount'].values[0])
print('Predicted fare amount:', fare_pred[0])


# Save the trained model to a file
joblib.dump(model, 'taxi_fare_model.pkl')

# Download the trained model file
