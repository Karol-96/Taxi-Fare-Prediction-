import random

# Define the parameters for generating the data
num_rows = 10000
min_distance = 0.5   # miles
max_distance = 10.0  # miles
min_time = 5         # minutes
max_time = 60        # minutes
min_passengers = 1
max_passengers = 4
payment_types = ['cash', 'credit card']
temp_range = (15, 30)  # degrees Celsius
weather_conditions = ['clear', 'cloudy', 'rainy']
traffic_conditions = ['light', 'moderate', 'heavy']
locations = ['Kathmandu', 'Lalitpur', 'Bhaktapur', 'Patan', 'Pokhara', 'Chitwan', 'Biratnagar', 'Birgunj', 'Dharan', 'Butwal']

# Define the function to generate a fare amount based on the other columns
def generate_fare(distance, time, passengers, payment_type, temperature, weather_condition, traffic_condition):
    base_fare = distance * 100 + time * 10 + passengers * 50
    if payment_type == 'credit card':
        base_fare *= 1.2
    if weather_condition == 'rainy':
        base_fare *= 1.1
    if traffic_condition == 'heavy':
        base_fare *= 1.5
    return int(base_fare + random.randint(-50, 50))

# Generate the data set
data = []
for i in range(num_rows):
    pickup = random.choice(locations)
    destination = random.choice(locations)
    while pickup == destination:
        destination = random.choice(locations)
    distance = round(random.uniform(min_distance, max_distance), 1)
    time = random.randint(min_time, max_time)
    passengers = random.randint(min_passengers, max_passengers)
    payment_type = random.choice(payment_types)
    temperature = random.randint(*temp_range)
    weather_condition = random.choice(weather_conditions)
    traffic_condition = random.choice(traffic_conditions)
    fare_amount = generate_fare(distance, time, passengers, payment_type, temperature, weather_condition, traffic_condition)
    row = (pickup, destination, distance, time, passengers, payment_type, temperature, weather_condition, traffic_condition, fare_amount)
    data.append(row)

# Save the data set to a CSV file
import csv
with open('taxi_fare_data_with_location.csv', 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['pickup', 'destination', 'distance', 'time', 'passengers', 'payment_type', 'temperature', 'weather_condition', 'traffic_condition', 'fare_amount'])
    writer.writerows(data)
