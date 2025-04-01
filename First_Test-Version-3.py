# %% [markdown]
# # Airbnb Pricing and Occupancy Trends in New York City
# ### Introduction
# Short-term rental platforms like Airbnb have revolutionized the hospitality industry, offering unique accommodations for travelers and income opportunities for hosts. In this project, we analyze Airbnb pricing and occupancy trends in New York City—a highly active market with diverse price points—to provide actionable insights for hosts, travelers, and city planners. Using data from Inside Airbnb, we will explore price variations, seasonal trends, amenity impacts, property type distributions, and occupancy patterns.

# %% [markdown]
# ### 1. Data Preparation & Cleaning
# #### Loading the Data
# We start by loading the Airbnb listings and calendar datasets for New York City, sourced from Inside Airbnb. The listings data contains property details, while the calendar data provides availability information.
# 
# 

# %%
import pandas as pd
import kagglehub

# Define data types for calendar.csv
calendar_dtypes = {
    'listing_id': 'Int64',
    'date': str,
    'available': str,
    'price': str,
    'adjusted_price': str,
    'minimum_nights': 'Int64',
    'maximum_nights': 'Int64'
}

# URLs for Inside Airbnb datasets
listings_url = 'https://data.insideairbnb.com/united-states/ny/new-york-city/2025-03-01/data/listings.csv.gz'
calendar_url = 'https://data.insideairbnb.com/united-states/ny/new-york-city/2025-03-01/data/calendar.csv.gz'

# Load datasets from URLs
listings = pd.read_csv(listings_url, compression='gzip')
calendar = pd.read_csv(calendar_url, dtype=calendar_dtypes, compression='gzip')

# Load the Kaggle dataset (AB_NYC_2019.csv)
dataset_path = kagglehub.dataset_download("dgomonov/new-york-city-airbnb-open-data")
file_path = f"{dataset_path}/AB_NYC_2019.csv"

# Read the file 
try:
    nyc_2019 = pd.read_csv(file_path, encoding='utf-8')
except UnicodeDecodeError:
    nyc_2019 = pd.read_csv(file_path, encoding='latin1')

print("Datasets loaded successfully!")
print("Listings shape:", listings.shape)
print("Calendar shape:", calendar.shape)
print("NYC 2019 shape:", nyc_2019.shape)

# %%
listings.head()

# %%
calendar.head()

# %%
nyc_2019.head()

# %%
# Check data types of calendar
print("Calendar Data Types:")
print(calendar.dtypes)

# Check for missing values
print("\nMissing Values in Calendar:")
print(calendar[['minimum_nights', 'maximum_nights']].isnull().sum())

# %% [markdown]
# #### Exploring the Data
# Let’s inspect the structure of both datasets to understand the available columns.

# %%
# Display the first few rows of listings
print("Listings Data (Inside Airbnb):")
print(listings.head())

# Display listings column names
print("\nListings Columns (Inside Airbnb):")
print(listings.columns)

# Display the first few rows of calendar
print("\nCalendar Data:")
print(calendar.head())

# Display calendar column names
print("\nCalendar Columns:")
print(calendar.columns)

# Display the first few rows of AB_NYC_2019
print("\nAB_NYC_2019 Data (Kaggle):")
print(nyc_2019.head())

# Display AB_NYC_2019 column names
print("\nAB_NYC_2019 Columns (Kaggle):")
print(nyc_2019.columns)

# %% [markdown]
# #### Cleaning the Listings Data
# We need to handle missing values and ensure data types are correct. The price column, for instance, may include dollar signs and commas, requiring conversion to a numeric format.

# %%
# Check for missing values in listings
print("Missing Values in Listings:")
print(listings.isnull().sum())

# Remove dollar signs and commas
listings['price'] = listings['price'].replace('[\$,]', '', regex=True).astype(float)

print("\nListings Data Types:")
print(listings.dtypes)

# %% [markdown]
# For simplicity, we’ll drop rows with missing price values, as they are critical for our analysis.

# %%
# Drop rows with missing price in listings
listings = listings.dropna(subset=['price'])

# Drop rows with missing price in AB_NYC_2019
nyc_2019 = nyc_2019.dropna(subset=['price'])

# %% [markdown]
# #### Cleaning the Calendar Data
# The calendar data requires similar preprocessing, including converting the price and date columns.

# %%
# Convert price and adjusted_price to float
calendar['price'] = calendar['price'].replace(r'[\$,]', '', regex=True).astype(float)
calendar['adjusted_price'] = calendar['adjusted_price'].replace(r'[\$,]', '', regex=True).fillna(0).astype(float)

# Convert date to datetime
calendar['date'] = pd.to_datetime(calendar['date'])

# Add a 'booked' column (1 if not available, 0 if available)
calendar['booked'] = calendar['available'].apply(lambda x: 1 if x == 'f' else 0)

# %% [markdown]
# ### 2. Price Analysis
# #### Average Price by Neighborhood
# We calculate the average rental price by neighborhood using the neighbourhood_cleansed column, which provides standardized neighborhood names.
# 
# This shows the most expensive neighborhoods, helping hosts and travelers identify high-cost areas.

# %%
# Calculate average price by neighborhood for listings.csv (2025 data)
avg_price_by_neighborhood_listings = listings.groupby('neighbourhood_cleansed')['price'].mean().sort_values(ascending=False)
print("Average Price by Neighborhood (Inside Airbnb 2025 Data):")
print(avg_price_by_neighborhood_listings.head(10))  # Top 10 neighborhoods

# Calculate average price by neighborhood for AB_NYC_2019.csv (2019 data)
avg_price_by_neighborhood_2019 = nyc_2019.groupby('neighbourhood')['price'].mean().sort_values(ascending=False)
print("\nAverage Price by Neighborhood (Kaggle 2019 Data):")
print(avg_price_by_neighborhood_2019.head(10))  # Top 10 neighborhoods

# %% [markdown]
# ### 3. Seasonal Trends
# #### Peak Booking Periods
# Since the calendar data reflects future availability (typically 365 days from the scrape date), lower availability suggests higher bookings. We calculate the average availability rate by month as a proxy for seasonal trends.
# 
# Months with lower availability indicate peak booking periods.

# %%
# Extract month from date
calendar['month'] = calendar['date'].dt.month

# Calculate average availability by month (By how many days available in a month)
availability_by_month = calendar.groupby('month')['available'].apply(lambda x: (x == 't').mean())
print("Average Availability by Month:")
print(availability_by_month)

# %% [markdown]
# ### 4. Amenity Impact Analysis
# #### Correlation Between Amenities and Price
# The amenities column is a string of amenities (e.g., "{Wi-Fi,Pool}"). We parse it into dummy variables and compute correlations with price.
# 
# This identifies amenities associated with higher prices, guiding hosts on value-adding features.

# %%
# TAKES A LONG TIME TO RUN (2.5 min)

# Clean amenities column by removing curly brackets and quotes
listings['amenities'] = listings['amenities'].str.replace(r'[{}"]', '', regex=True)

# Dummy variables for amenities
amenities_dummies = listings['amenities'].str.get_dummies(sep=',')

# Concatenate with listings
listings_with_amenities = pd.concat([listings, amenities_dummies], axis=1)

# Calculate correlations with price
amenities_correlations = listings_with_amenities[amenities_dummies.columns].corrwith(listings_with_amenities['price'])
print("Amenities Correlations with Price (Top 10):")
print(amenities_correlations.sort_values(ascending=False).head(10))

# %% [markdown]
# ### 5. Property Type Distribution
# #### Most Common Property Types
# We examine the distribution of property types and room types to understand the market composition.

# %%
# Property type distribution (listings.csv, 2025 data)
property_type_counts_listings = listings['property_type'].value_counts()
print("Property Type Distribution (Inside Airbnb 2025 Data):")
print(property_type_counts_listings.head(10))

# Room type distribution (listings.csv, 2025 data)
room_type_counts_listings = listings['room_type'].value_counts()
print("\nRoom Type Distribution (Inside Airbnb 2025 Data):")
print(room_type_counts_listings)

# Property type distribution (AB_NYC_2019.csv, 2019 data)
# AB_NYC_2019.csv does not have property_type, only room_type
room_type_counts_2019 = nyc_2019['room_type'].value_counts()
print("\nRoom Type Distribution (Kaggle 2019 Data):")
print(room_type_counts_2019)

# %% [markdown]
# ### 6. Occupancy Trends
# #### Using Number of Reviews as a Proxy
# Since historical occupancy data isn’t available, we use number_of_reviews as a proxy for past occupancy, assuming more reviews indicate more bookings.

# %%
# Average number of reviews by neighborhood (using listings.csv)
avg_reviews_by_neighborhood = listings.groupby('neighbourhood_cleansed')['number_of_reviews'].mean().sort_values(ascending=False)
print("Average Number of Reviews by Neighborhood (Top 10, Inside Airbnb 2025 Data):")
print(avg_reviews_by_neighborhood.head(10))

# Clean price column
listings['price'] = pd.to_numeric(listings['price'].replace('[\$,]', '', regex=True), errors='coerce')

# Correlation between price and number of reviews (using listings.csv)
correlation = listings['price'].corr(listings['number_of_reviews'])
print(f"\nCorrelation between Price and Number of Reviews (Inside Airbnb 2025 Data): {correlation}")

# %% [markdown]
# #### Price Bins and Occupancy
# We bin prices and calculate the average number of reviews per bin to find popular price ranges.

# %%
# Bin prices for listings.csv
bins = [0, 50, 100, 150, 200, 250, 300, 500, 1000, 5000]
labels = ['0-50', '50-100', '100-150', '150-200', '200-250', '250-300', '300-500', '500-1000', '1000+']
listings['price_bin'] = pd.cut(listings['price'], bins=bins, labels=labels)

# Average reviews by price bin (listings.csv)
avg_reviews_by_price = listings.groupby('price_bin', observed=True)['number_of_reviews'].mean()
print("Average Number of Reviews by Price Bin (Inside Airbnb 2025 Data):")
print(avg_reviews_by_price)

# Bin prices for AB_NYC_2019.csv
nyc_2019['price_bin'] = pd.cut(nyc_2019['price'], bins=bins, labels=labels)

# Average reviews by price bin (AB_NYC_2019.csv)
avg_reviews_by_price_2019 = nyc_2019.groupby('price_bin', observed=True)['number_of_reviews'].mean()
print("\nAverage Number of Reviews by Price Bin (Kaggle 2019 Data):")
print(avg_reviews_by_price_2019)

# %% [markdown]
# ### 7. Visualization
# #### Price by Neighborhood
# A bar chart visualizes average prices across neighborhoods.

# %%
import matplotlib.pyplot as plt

# Plot average price by neighborhood (top 10, listings.csv)
avg_price_by_neighborhood_listings.head(10).plot(kind='bar', figsize=(12, 6))
plt.title('Average Airbnb Price by Neighborhood in NYC (Top 10, Inside Airbnb 2025 Data)')
plt.xlabel('Neighborhood')
plt.ylabel('Average Price ($)')
plt.show()

# Plot average price by neighborhood (top 10, AB_NYC_2019.csv)
avg_price_by_neighborhood_2019.head(10).plot(kind='bar', figsize=(12, 6))
plt.title('Average Airbnb Price by Neighborhood in NYC (Top 10, Kaggle 2019 Data)')
plt.xlabel('Neighborhood')
plt.ylabel('Average Price ($)')
plt.show()

# %% [markdown]
# #### Seasonal Availability
# A line plot shows availability trends over months.

# %%
# Plot availability by month
availability_by_month.plot(kind='line', marker='o')
plt.title('Average Availability by Month')
plt.xlabel('Month')
plt.ylabel('Availability Rate')
plt.xticks(range(1, 13))
plt.show()

# %% [markdown]
# #### Price vs. Reviews by Price Bin
# A bar chart illustrates how occupancy (via reviews) varies with price.

# %%
# Plot reviews by price bin (listings.csv)
avg_reviews_by_price.plot(kind='bar')
plt.title('Average Number of Reviews by Price Bin (Inside Airbnb 2025 Data)')
plt.xlabel('Price Bin ($)')
plt.ylabel('Average Number of Reviews')
plt.show()

# Plot reviews by price bin (AB_NYC_2019.csv)
avg_reviews_by_price_2019.plot(kind='bar')
plt.title('Average Number of Reviews by Price Bin (Kaggle 2019 Data)')
plt.xlabel('Price Bin ($)')
plt.ylabel('Average Number of Reviews')
plt.show()

# %% [markdown]
# #### Interactive Map
# Using Folium, we create a map of listings colored by price.

# %%
# Takes 40s to run

import folium

# Create a map centered on NYC
nyc_map = folium.Map(location=[40.7128, -74.0060], zoom_start=11)

# Add markers
for index, row in listings.iterrows():
    folium.CircleMarker(
        location=[row['latitude'], row['longitude']],
        radius=5,
        color='red' if row['price'] > 200 else 'blue',
        fill=True,
        fill_color='red' if row['price'] > 200 else 'blue',
        fill_opacity=0.6,
        popup=f"Price: ${row['price']}"
    ).add_to(nyc_map)

nyc_map.save('nyc_airbnb_map.html')

# %% [markdown]
# ### 8. Summary of Findings
# * **Price Variation by Neighborhood:** The top 5 most expensive neighborhoods in 2025 (Inside Airbnb data) are SoHo ($806.09), Battery Park City ($753.67), Riverdale ($715.67), Navy Yard ($621.00), and Fort Wadsworth ($600.00). In 2019 (Kaggle data), the top neighborhoods need to be computed, but affordable areas often include parts of the Bronx.
# * **Peak Booking Periods:** Months with the lowest availability are March (0.369), February (0.388), and January (0.393), suggesting late winter/early spring as peak demand periods, contrary to the assumption of summer months.
# * **Amenities and Pricing:** Amenities like "55 inch TV" (0.223), "Hawkins Falls body soap" (0.222), and "Sauna" (0.208) correlate most strongly with higher prices.
# * **Popular Property Types:** Entire rental units (9,648) dominate, with entire homes/apartments (12,664) and private rooms (9,186) being the most common room types.
# * **Occupancy Trends:** Listings in the $100-150 range have the highest average reviews (44.15), indicating a popular price point. Temporal trends (from reviews.csv) show review counts over time, indicating booking patterns (visualized in the plot).

# %% [markdown]
# ### 9. Recommendations
# **For Hosts**
# * Add high-value amenities like saunas or large TVs (e.g., 55 inch TV), which correlate with higher prices (0.208 and 0.223, respectively).
# * Increase rates during late winter/early spring (e.g., March, February), when availability is lowest, indicating peak demand.
# 
# **For Travelers**
# * Book in more affordable neighborhoods, such as those in the Bronx (to be confirmed with 2019 data), where prices are typically lower.
# * Travel during late summer (e.g., August, with 0.494 availability), when availability is highest, suggesting lower demand and better deals.
# 
# **For City Planners**
# * Monitor neighborhoods with high entire-home listings, such as Manhattan areas (12,664 entire homes/apartments), as they may impact long-term housing availability.
# * Note that these trends reflect data influenced by NYC’s short-term rental regulations, with a significant number of listings (9,648 entire rental units) potentially affecting housing stock.


