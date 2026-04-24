# ================================
# TASK 4 - LOCATION BASED ANALYSIS
# ================================

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
file_path = "Dataset.csv"   # Change to your file name if different
df = pd.read_csv(file_path)

# -------------------------------
# 1️⃣ Basic Dataset Info
# -------------------------------
print("\nDataset Shape:", df.shape)
print("\nColumns in Dataset:\n", df.columns)

# -------------------------------
# 2️⃣ Restaurant Distribution by City
# -------------------------------
city_counts = df['City'].value_counts()

print("\nTop 10 Cities with Most Restaurants:")
print(city_counts.head(10))

plt.figure(figsize=(10,6))
city_counts.head(10).plot(kind='bar')
plt.title("Top 10 Cities by Number of Restaurants")
plt.xlabel("City")
plt.ylabel("Number of Restaurants")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# -------------------------------
# 3️⃣ Average Rating by City
# -------------------------------
avg_rating_city = df.groupby('City')['Aggregate rating'].mean().sort_values(ascending=False)

print("\nTop 10 Cities by Average Rating:")
print(avg_rating_city.head(10))

plt.figure(figsize=(10,6))
avg_rating_city.head(10).plot(kind='bar', color='green')
plt.title("Top 10 Cities by Average Rating")
plt.xlabel("City")
plt.ylabel("Average Rating")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# -------------------------------
# 4️⃣ Restaurant Location Map (Scatter Plot)
# -------------------------------
plt.figure(figsize=(8,6))
sns.scatterplot(
    x=df['Longitude'],
    y=df['Latitude'],
    hue=df['Aggregate rating'],
    palette='coolwarm'
)

plt.title("Restaurant Locations (Latitude vs Longitude)")
plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.legend(title="Rating")
plt.show()

# -------------------------------
# 5️⃣ Average Cost by City
# -------------------------------
if 'Average Cost for two' in df.columns:
    avg_cost = df.groupby('City')['Average Cost for two'].mean().sort_values(ascending=False)
    
    print("\nTop 10 Cities by Average Cost for Two:")
    print(avg_cost.head(10))
    
    plt.figure(figsize=(10,6))
    avg_cost.head(10).plot(kind='bar', color='orange')
    plt.title("Top 10 Cities by Average Cost for Two")
    plt.xlabel("City")
    plt.ylabel("Average Cost")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

print("\n✅ Task 4 Completed Successfully!")