# assignment.py

import pandas as pd
import matplotlib.pyplot as plt

# -----------------------------
# Task 1: Load and Explore the Dataset
# -----------------------------

# Load dataset directly from seaborn's built-in link (or you can download iris.csv and use it locally)
url = "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/iris.csv"
df = pd.read_csv(url)

# Display first 5 rows
print("First 5 rows of the dataset:")
print(df.head())

# Check data types and missing values
print("\nDataset info:")
print(df.info())

print("\nMissing values in each column:")
print(df.isnull().sum())

# Clean dataset (if any missing values existed)
df = df.dropna()

# -----------------------------
# Task 2: Basic Data Analysis
# -----------------------------

# Basic statistics
print("\nBasic statistics:")
print(df.describe())

# Group by species and get average of numerical columns
grouped = df.groupby("species").mean(numeric_only=True)
print("\nAverage values by species:")
print(grouped)

# Identify patterns (just printing observations)
print("\nObservations:")
print("-> Setosa has the smallest petal and sepal sizes.")
print("-> Virginica has the largest petal and sepal sizes on average.")
print("-> Versicolor lies between the two.")

# -----------------------------
# Task 3: Data Visualization
# -----------------------------

# 1. Line Chart: trend of sepal_length across dataset index
plt.figure(figsize=(8,5))
plt.plot(df.index, df['sepal_length'], color='blue', label='Sepal Length')
plt.title('Trend of Sepal Length Across Samples')
plt.xlabel('Sample Index')
plt.ylabel('Sepal Length (cm)')
plt.legend()
plt.tight_layout()
plt.show()

# 2. Bar Chart: average petal_length per species
plt.figure(figsize=(6,4))
grouped['petal_length'].plot(kind='bar', color=['red','green','blue'])
plt.title('Average Petal Length by Species')
plt.xlabel('Species')
plt.ylabel('Petal Length (cm)')
plt.tight_layout()
plt.show()

# 3. Histogram: distribution of sepal_width
plt.figure(figsize=(6,4))
plt.hist(df['sepal_width'], bins=15, color='purple', edgecolor='black')
plt.title('Distribution of Sepal Width')
plt.xlabel('Sepal Width (cm)')
plt.ylabel('Frequency')
plt.tight_layout()
plt.show()

# 4. Scatter Plot: sepal_length vs petal_length colored by species
plt.figure(figsize=(6,4))
for species in df['species'].unique():
    subset = df[df['species'] == species]
    plt.scatter(subset['sepal_length'], subset['petal_length'], label=species)
plt.title('Sepal Length vs Petal Length by Species')
plt.xlabel('Sepal Length (cm)')
plt.ylabel('Petal Length (cm)')
plt.legend()
plt.tight_layout()
plt.show()
