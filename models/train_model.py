import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import pickle

# Load data
df = pd.read_csv('../data/car_data.csv')

# Encode manually (IMPORTANT)
df['Fuel_Type'] = df['Fuel_Type'].map({'Petrol':1, 'Diesel':0})
df['Seller_Type'] = df['Seller_Type'].map({'Individual':1, 'Dealer':0})
df['Transmission'] = df['Transmission'].map({'Manual':1, 'Automatic':0})

# Feature engineering
df['car_age'] = 2024 - df['Year']

# Drop unused
df.drop(['Year', 'Car_Name'], axis=1, inplace=True, errors='ignore')

# Features
X = df[['Present_Price','Kms_Driven','Owner','Fuel_Type','Seller_Type','Transmission','car_age']]
y = df['Selling_Price']

# Train
model = RandomForestRegressor()
model.fit(X, y)

# Save
pickle.dump(model, open('model.pkl', 'wb'))

print("✅ Model trained with 7 features!")