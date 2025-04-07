#always the same pandas and shit
import pandas as pd
import numpy as np
import requests
import matplotlib.pyplot as plt

from datetime import datetime
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, precision_score, recall_score, f1_score
from sklearn.impute import KNNImputer

#kasper
url = 'https://raw.githubusercontent.com/KasperLoos/ConsultElement/main/data/bronze_data/startup_failures.csv'
response = requests.get(url)

# Save the content of the file
with open('startup_failures.csv', 'wb') as file:
    file.write(response.content)

# Read the CSV file into a pandas DataFrame
df = pd.read_csv('startup_failures.csv')

# Check if file exists just to be sure (it will print TRUE)
import os
print(os.path.exists('startup_failures.csv'))
with open('startup_failures.csv', 'wb') as file:
    file.write(response.content)

# Show the first 5 rows just to check
df = pd.read_csv('startup_failures.csv')
print(df.head())  

#basic info about data
print(df.info())
# Count of values for each column
print(df.nunique())

#load data
url = 'https://raw.githubusercontent.com/KasperLoos/ConsultElement/refs/heads/main/data/bronze_data/startup_failures.csv'
startup = pd.read_csv(url)
startup['loan_approval'] = startup['status'].map({'operating':1, 'acquired' : 1, 'ipo' : 1, 'closed' : 0 })
startup['funding_total_usd'] = pd.to_numeric(startup['funding_total_usd'], errors='coerce')
startup['funding_total_usd'].fillna(0, inplace = True)

# Convert date columns
startup['founded_at'] = pd.to_datetime(startup['founded_at'], errors='coerce')
startup['first_funding_at'] = pd.to_datetime(startup['first_funding_at'], errors='coerce')
startup['last_funding_at'] = pd.to_datetime(startup['last_funding_at'], errors = 'coerce')

#drop missing values + weird dates -> less than 5% of data in this column
startup = startup.dropna(subset=['first_funding_at']).reset_index(drop=True)
start_date = '1970-01-01'
end_date = '2016-01-01'
startup = startup[startup['last_funding_at'].between(start_date, end_date)].reset_index(drop = True)

#convert to meaningfull columns
startup['founded_year'] = startup['founded_at'].dt.year
startup['founded_month'] = startup['founded_at'].dt.month
startup['funding_duration'] = (startup['last_funding_at'] - startup['first_funding_at']).dt.days
startup['funding_delay'] = (startup['first_funding_at'] - startup['founded_at']).dt.days
startup['mean_funding_per_round'] = startup['funding_total_usd'] / startup['funding_rounds']    

# Fill missing values with values of first funding date
startup['founded_year'] = startup['founded_year'].fillna(startup['first_funding_at'].dt.year)
startup['founded_month'] = startup['founded_month'].fillna(startup['first_funding_at'].dt.month)

# Impute/fill missing values funding_delay based on both country_code and founded_year
startup['funding_delay'] = startup.groupby(['country_code', 'founded_year'])['funding_delay'].transform(
    lambda x: x.fillna(x.median())  # Replace median with mean if you prefer)
)

# Now, let's fill missing 'state_code' for non-USA companies with NOTUSA
df.loc[(df['state_code'].isnull()) & (df['country_code'] != 'USA'), 'state_code'] = 'NOTUSA'
# For USA-based companies (if needed), you can fill with default 'STATE' just to fill it in and nothave a missing value
df.loc[(df['state_code'].isnull()) & (df['country_code'] == 'USA'), 'state_code'] = 'STATE'

# If country is missing, fill it by examining since we have the information of the region or city.
# If region is missing, fill in with its country.
# If city is missing, fill it in with the country.
from geopy.geocoders import Nominatim
geolocator = Nominatim(user_agent="geoapiExercises")

def geocode(location):
    try:
        if location:  # Only try to geocode if location is not empty
            location_info = geolocator.geocode(location)
            if location_info:
                # Return the last part of the address (usually the country)
                return location_info.address.split(', ')[-1]
            else:
                return None
        return None
    except Exception as e:
        print(f"Error during geocoding: {e}")
        return None

# 1. If country is missing, fill it by examining region or city using geocoding
startup['country_code'] = startup.apply(
    lambda row: geocode(row['region']) if pd.isna(row['country_code']) and pd.notna(row['region']) else
               (geocode(row['city']) if pd.isna(row['country_code']) and pd.notna(row['city']) else row['country_code']), axis=1)

# 2. If region is missing, fill it with the country using geocoding
startup['region'] = startup.apply(
    lambda row: row['country_code'] if pd.isna(row['region']) and pd.notna(row['country_code']) else row['region'], axis=1)

# 3. If city is missing, fill it with the country using geocoding
startup['city'] = startup.apply(
    lambda row: row['country_code'] if pd.isna(row['city']) and pd.notna(row['country_code']) else row['city'], axis=1)

# Check if there are any missing values in the 'country_code', 'region', and 'city' columns
missing_values = startup[['country_code', 'region', 'city']].isna().sum()
print(missing_values) #there are 6995 rows for which no country, no region no city is available
#I am going to do webscraping

import requests
from bs4 import BeautifulSoup
import re

# Function to extract country code from domain
def extract_country_from_url(url):
    if pd.isna(url):
        return None
    match = re.search(r"\.([a-z]{2})$", url)
    if match:
        return match.group(1).upper()
    return None
# Apply function to extract country codes, only if the value is missing (NaN)
startup['country_code'] = startup.apply(
    lambda row: extract_country_from_url(row['homepage_url']) if pd.isna(row['country_code']) else row['country_code'], 
    axis=1
)
print(startup[['homepage_url', 'country_code']].head())

# Check for missing values in 'country_code', 'region', and 'city' columns
missing_values = startup[['country_code', 'region', 'city']].isna().sum()
print(missing_values)
# Fill missing 'region' with 'country_code', only if 'region' is missing
startup.loc[startup['region'].isna(), 'region'] = startup['country_code']
# Fill missing 'city' with 'country_code', only if 'city' is missing
startup.loc[startup['city'].isna(), 'city'] = startup['country_code']
print("Missing values in 'region' after fill:", startup['region'].isna().sum())
print("Missing values in 'city' after fill:", startup['city'].isna().sum())
#it worked

#now fill state code with either its usa or not usa
startup['state_code'] = startup.apply(
    lambda row: 'USA' if row['country_code'] == 'USA' and pd.isna(row['state_code']) else 
                ('NOTUSA' if pd.isna(row['state_code']) else row['state_code']),
    axis=1
)

# Check missing values for all columns
missing_values = startup.isna().sum()
print(missing_values)
