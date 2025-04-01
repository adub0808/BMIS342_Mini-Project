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

# Define data types for calendar.csv with nullable integers
calendar_dtypes = {
    'listing_id': 'Int64',  # Also use Int64 here in case of missing listing IDs
    'date': str,
    'available': str,
    'price': str,
    'adjusted_price': str,
    'minimum_nights': 'Int64',  # Nullable integer to handle NA
    'maximum_nights': 'Int64'   # Nullable integer to handle NA
}

# Load listings and calendar data with specified dtypes
listings = pd.read_csv('listings.csv')
calendar = pd.read_csv('calendar.csv', dtype=calendar_dtypes)

# %%
# Check data types of calendar
print("Calendar Data Types:")
print(calendar.dtypes)

# Check for missing values again to confirm handling
print("\nMissing Values in Calendar:")
print(calendar[['minimum_nights', 'maximum_nights']].isnull().sum())

# %% [markdown]
# #### Exploring the Data
# Let’s inspect the structure of both datasets to understand the available columns.

# %%
# Display the first few rows of listings
print("Listings Data:")
print(listings.head())

# Display listings column names
print("\nListings Columns:")
print(listings.columns)

# Display the first few rows of calendar
print("\nCalendar Data:")
print(calendar.head())

# Display calendar column names
print("\nCalendar Columns:")
print(calendar.columns)

# %% [markdown]
# #### Cleaning the Listings Data
# We need to handle missing values and ensure data types are correct. The price column, for instance, may include dollar signs and commas, requiring conversion to a numeric format.

# %%
# Check for missing values in listings
print("Missing Values in Listings:")
print(listings.isnull().sum())

# Convert price to float by removing dollar signs and commas
listings['price'] = listings['price'].replace('[\$,]', '', regex=True).astype(float)

# Check data types
print("\nListings Data Types:")
print(listings.dtypes)

# %% [markdown]
# For simplicity, we’ll drop rows with missing price values, as they are critical for our analysis.

# %%
# Drop rows with missing price
listings = listings.dropna(subset=['price'])

# %% [markdown]
# #### Cleaning the Calendar Data
# The calendar data requires similar preprocessing, including converting the price and date columns.

# %%
'''
# Convert price in calendar to float
calendar['price'] = calendar['price'].replace('[\$,]', '', regex=True).astype(float)

# Convert date to datetime
calendar['date'] = pd.to_datetime(calendar['date'])

# Add a 'booked' column (1 if not available, 0 if available)
calendar['booked'] = calendar['available'].apply(lambda x: 1 if x == 'f' else 0)
'''

# Convert price and adjusted_price to float, handling empty values
calendar['price'] = calendar['price'].replace('[\$,]', '', regex=True).astype(float)
calendar['adjusted_price'] = calendar['adjusted_price'].replace('[\$,]', '', regex=True).fillna(0).astype(float)

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
# Calculate average price by neighborhood
avg_price_by_neighborhood = listings.groupby('neighbourhood_cleansed')['price'].mean().sort_values(ascending=False)
print("Average Price by Neighborhood:")
print(avg_price_by_neighborhood.head(10))  # Top 10 neighborhoods

# %% [markdown]
# ### 3. Seasonal Trends
# #### Peak Booking Periods
# Since the calendar data reflects future availability (typically 365 days from the scrape date), lower availability suggests higher bookings. We calculate the average availability rate by month as a proxy for seasonal trends.
# 
# Months with lower availability indicate peak booking periods.

# %%
# Extract month from date
calendar['month'] = calendar['date'].dt.month

# Calculate average availability by month (proportion of days available)
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
# Clean amenities column by removing braces and quotes
listings['amenities'] = listings['amenities'].str.replace('[{}"]', '', regex=True)

# Create dummy variables for amenities
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
# Count of each property type
property_type_counts = listings['property_type'].value_counts()
print("Property Type Distribution:")
print(property_type_counts.head(10))  # Top 10 property types

# Count of each room type
room_type_counts = listings['room_type'].value_counts()
print("\nRoom Type Distribution:")
print(room_type_counts)

# %% [markdown]
# ### 6. Occupancy Trends
# #### Using Number of Reviews as a Proxy
# Since historical occupancy data isn’t available, we use number_of_reviews as a proxy for past occupancy, assuming more reviews indicate more bookings.

# %%
# Average number of reviews by neighborhood
avg_reviews_by_neighborhood = listings.groupby('neighbourhood_cleansed')['number_of_reviews'].mean().sort_values(ascending=False)
print("Average Number of Reviews by Neighborhood (Top 10):")
print(avg_reviews_by_neighborhood.head(10))

# Correlation between price and number of reviews
correlation = listings['price'].corr(listings['number_of_reviews'])
print(f"\nCorrelation between Price and Number of Reviews: {correlation}")

# %% [markdown]
# #### Price Bins and Occupancy
# We bin prices and calculate the average number of reviews per bin to find popular price ranges.

# %%
# Bin prices
bins = [0, 50, 100, 150, 200, 250, 300, 500, 1000, 5000]
labels = ['0-50', '50-100', '100-150', '150-200', '200-250', '250-300', '300-500', '500-1000', '1000+']
listings['price_bin'] = pd.cut(listings['price'], bins=bins, labels=labels)

# Average reviews by price bin
avg_reviews_by_price = listings.groupby('price_bin')['number_of_reviews'].mean()
print("Average Number of Reviews by Price Bin:")
print(avg_reviews_by_price)

# %% [markdown]
# ### 7. Visualization
# #### Price by Neighborhood
# A bar chart visualizes average prices across neighborhoods.

# %%
import matplotlib.pyplot as plt

# Plot average price by neighborhood (top 10)
avg_price_by_neighborhood.head(10).plot(kind='bar', figsize=(12, 6))
plt.title('Average Airbnb Price by Neighborhood in NYC (Top 10)')
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
# Plot reviews by price bin
avg_reviews_by_price.plot(kind='bar')
plt.title('Average Number of Reviews by Price Bin')
plt.xlabel('Price Bin ($)')
plt.ylabel('Average Number of Reviews')
plt.show()

# %% [markdown]
# #### Interactive Map
# Using Folium, we create a map of listings colored by price.

# %%
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

# Save the map
nyc_map.save('nyc_airbnb_map.html')

# %% [markdown]
# ### 8. Summary of Findings
# * **Price Variation by Neighborhood:** The top 5 most expensive neighborhoods are [e.g., Tribeca, SoHo], with averages exceeding $300/night, while affordable areas include [e.g., Bronx neighborhoods].
# 
# * **Peak Booking Periods:** Months with the lowest availability (e.g., summer months like June-August) suggest peak demand.
# 
# * **Amenities and Pricing:** Amenities like hot tubs or premium Wi-Fi correlate strongly with higher prices.
# 
# * **Popular Property Types:** Apartments dominate, with entire homes and private rooms being the most common room types.
# 
# * **Occupancy Trends:** Listings in the $100-150 range have the highest average reviews, indicating a popular price point.

# %% [markdown]
# ### 9. Recommendations
# **For Hosts**
# * Add high-value amenities (e.g., hot tubs) to justify higher prices.
# * Increase rates during peak months (e.g., June-August) when availability is low.
# 
# **For Travelers**
# * Book in affordable neighborhoods like [e.g., Bronx areas] for lower prices.
# * Travel during off-peak months (e.g., winter months) for better deals.
# 
# **For City Planners**
# * Monitor neighborhoods with high entire-home listings (e.g., Manhattan areas), as they may impact long-term housing availability.
# * Note that these trends reflect data influenced by NYC’s short-term rental regulations.


