# Google_Play_Store_Apps_rating_Prediction_p1
## Project Overview
#### Project Title: Google Play Store Apps Rating Prediction
#### Level: Advance
#### Tools: Visual Studio/Jupyter Notebook, Excel, Postgre SQL, Tableau
#### Languages: Python, SQL
#### Database`google_paly_store-1.xlsx`

This project is designed to demonstrate Excel, python, SQL, Tableau skills and techniques typically used by data analysts to explore, clean, and analyze google play store apps rating data. The project involves setting up a google paly store database, performing exploratory data analysis (EDA), visualization, machine learning techniques and answering specific business questions through python, SQL queries. This project is ideal for those who are starting their journey in data analysis and want to build a solid foundation in python, SQL, tableau.

## Objectives
1. Set up a google play store database: Create and predicting google play store apps rating database with the provided google play store data.
2. Data Cleaning: Identify and remove any records with missing or null values.
3. Exploratory Data Analysis (EDA): Perform basic exploratory data analysis to understand the dataset.
4. Finance Analysis: Use python & SQL to answer specific business questions and derive insights from the google play sote data.

## 1. Python Project Structure
## About Dataset
### Context
While many public datasets (on Kaggle and the like) provide Apple App Store data, there are not many counterpart datasets available for Google Play Store apps anywhere on the web. On digging deeper, I found out that iTunes App Store page deploys a nicely indexed appendix-like structure to allow for simple and easy web scraping. On the other hand, Google Play Store uses sophisticated modern-day techniques (like dynamic page load) using JQuery making scraping more challenging.

### Content
Each app (row) has values for catergory, rating, size, and more.

### About data columns:
##### App : The name of the app
##### Category : The category of the app
##### Rating : The rating of the app in the Play Store
##### Reviews : The number of reviews of the app
##### Size : The size of the app
##### Install : The number of installs of the app
##### Type : The type of the app (Free/Paid)
##### Price : The price of the app (0 if it is Free)
##### Content Rating : The appropiate target audience of the app
##### Genres: The genre of the app
##### Last Updated : The date when the app was last updated
##### Current Ver : The current version of the app
##### Android Ver : The minimum Android version required to run the app

### 1. Import Modules Libraries
```python
import numpy as np  # Manipulation Numbers
import pandas as pd # Data Frames
import matplotlib.pyplot as plt #Visualisation
%matplotlib inline  
import seaborn as sns #Visualisation
import plotly.express as px
import warnings
warnings.filterwarnings("ignore")
```

### 2. Load and Explore The Data
```python
# Load the data
data = pd.read_csv(r'C:\Users\thamm\OneDrive\Documents\Internship\Projects\Google Playstore Apps Rating Prediction\googleplaystore.csv')

# Analyze the data
df = pd.DataFrame(data)
df

# Dimension of data
df.shape

# Basic information about the data
df.info()

# Rows
df.index

# Columns Names
df.columns

# Head of the data
df.head()

Tail of the data
df.tail()

#Check for datatypes
df.dtypes
```

### 3. Data Cleaning and Preprocessing

#### 1.Handling Missing Values: We will identify and handle missing values in the dataset.
```python
# Check for missing values for each column
data.isnull().sum()

# Drop rows with missing values in important columns
df.dropna(subset=['Type', 'Content Rating'], inplace=True)
df['Rating'].fillna(df['Rating'].median(), inplace=True)
df['Type'].fillna(df['Type'].mode()[0], inplace=True)
df['Content Rating'].fillna(df['Content Rating'].mode()[0], inplace=True)
df['Android Ver'].fillna(df['Android Ver'].mode()[0], inplace=True)
df['Current Ver'].fillna(df['Current Ver'].mode()[0], inplace=True)

# After handling missing values, recheck for any nulls:
df.isnull().sum()

# Check the updated data
df.info()
df.shape

# Remove Duplicates Rows
df.drop_duplicates(inplace=True)
```

### 3.1 Converting Columns to Appropriate Data Types:
#### *Convert Reviews and Installs to integer types.
#### *Convert Price to numeric.
#### *Convert Size to a uniform numeric format.

```python
# Convert 'Reviews' to integer
df['Reviews'] = df['Reviews'].astype(int)
df['Reviews']

# Convert 'Installs' by removing '+' and ',' then converting to integer
df['Installs'] = df['Installs'].apply(lambda x: str(x).replace(',', '').replace('+', '')).astype(int)
df['Installs']

# Convert 'Price' by removing '$' and converting to float
df['Price'] = df['Price'].apply(lambda x: str(x).replace('$', '')).astype(float)
df['Price']

# Convert 'Size' to numeric (MB) - Convert 'k' to MB
def convert_size(size):
    if 'M' in size:

        return float(size.replace('M', ''))
    elif 'k' in size:

        return float(size.replace('k', '')) / 1000
    else:

        return np.nan

df['Size'] = df['Size'].apply(convert_size)
df['Size']
```
### 4. Exploratory Data Analysis (EDA):
```python
# Category
data['Category'].unique()

data[data['Category'] == '1.9']

data['Category'].loc[10472]=np.nan
data['Category'].loc[10472]

df_category = data['Category'].value_counts()
df_category

plt.figure(figsize=(12,5))
sns.barplot(x=df_category.values, y=df_category.index, orient='h')
plt.title('Distribution of Category Type', fontsize=14)
plt.xlabel('No.of Apps', fontsize=12)
plt.ylabel('Category', fontsize=12)
plt.show()

# Distribution of App Rating
plt.figure(figsize=(12,5))
sns.histplot(df['Rating'].dropna(), bins=20, kde=True)
plt.title('Distribution of App Ratings', fontsize=14)
plt.xlabel('Rating', fontsize=12)
plt.ylabel('Count', fontsize=12)
plt.show()

# Top 10 Categories by Number of Apps
plt.figure(figsize=(12,5))
top_categories = df['Category'].value_counts().head(10)
sns.barplot(x=top_categories.index, y=top_categories.values, palette='coolwarm')
plt.title('Top 10 App Categories by Number of Apps', fontsize=14)
plt.ylabel('Number of Apps', fontsize=12)
plt.xlabel('Category', fontsize=12)
plt.xticks(rotation=45)
plt.show()

# Free vs Paid Apps
df['Type'].value_counts()

plt.figure(figsize=(12,8))
sns.countplot(data=df, x='Type', palette='Set2')

plt.title('Distribution of Free vs Paid Apps', fontsize=14)
plt.xlabel('Type', fontsize=12)
plt.ylabel('No.of Apps', fontsize=12)
plt.show()

plt.figure(figsize=(10,5))
df['Type'].value_counts().plot.pie(autopct = '%1.1f%%')

# Correlation Between reviews and Rating
plt.figure(figsize=(12,8))
sns.scatterplot(x='Reviews', y='Rating', data=df, hue='Category')
plt.title('Correlation Between Reviews and Ratings')
plt.legend()
plt.show()
```

### 4.1 Price Analysis
```python
# Price Distribution for Paid Apps
paid_apps = df[df['Type'] == 'Paid']
plt.figure(figsize=(10,5))
sns.histplot(paid_apps['Price'], bins=30, color='orange')
plt.title('Price Distribution for Paid Apps', fontsize=14)
plt.xlabel('Price ($)', fontsize=12)
plt.ylabel('Count', fontsize=12)
plt.show()

# Relationship between Price and Rating
plt.figure(figsize=(12,5))
sns.scatterplot(x='Price', y='Rating', data=paid_apps)
plt.title('Price vs Rating for Paid Apps', fontsize=14)
plt.show()
```

### 4.2 Content Rating Analysis
```python
# Distribution of Content Ratings
content_ratings = df['Content Rating'].value_counts()
plt.figure(figsize=(12,6))
sns.barplot(x=content_ratings.index, y=content_ratings.values, palette='coolwarm')
plt.title('Distribution of Content Ratings', fontsize=14)
plt.xlabel('Content Rating', fontsize=12)
plt.ylabel('Count', fontsize=12)
plt.show()

# Content Rating vs Rating
plt.figure(figsize=(10,6))
sns.boxplot(x='Content Rating', y='Rating', data=df, palette='Set1')
plt.title('Content Rating vs App Rating')
plt.show()
```

### 4.3 Genre and Installs Analysis
```python
# Top Genres by Install Count
top_genres_installs= df.groupby('Genres')['Installs'].sum().sort_values(ascending=False).head(10)
plt.figure(figsize=(12,5))
sns.barplot(x=top_genres_installs.index, y=top_genres_installs.values, palette='Spectral')
plt.xticks(rotation=90)
plt.title('Top 10 Genres by Install Count', fontsize=14)
plt.show()
```

### 5.Machine Learning(Predicting App Rating)
#### Prepare Data for Modwling
#### We will predict the app rating based on the features in the dataset. First, let's prepare the data by encoding categorical variables and splitting it into training and testing sets.
```python
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

label_encoder = LabelEncoder()
df['Category'] = label_encoder.fit_transform(df['Category'])
df['Type'] = label_encoder.fit_transform(df['Type'])
df['Content Rating'] = label_encoder.fit_transform(df['Content Rating'])
df['Genres'] = label_encoder.fit_transform(df['Genres'])

# Define features and target variable
X = df[['Category', 'Reviews', 'Size', 'Installs', 'Type', 'Price', 'Genres', 'Content Rating']]
y = df['Rating']

# Handle missing values in target
y.fillna(y.median(), inplace=True)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

X_train.head()

X_test.head()

y_test.head()

y_train.head()
```

#### Tarin a Random Forest Model
```python
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# Train the model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f'Mean Squared Error: {mse}')
print(f'R-squared: {r2}')
```

## 2. SQL Project Structure
### 1. Database Setup
##### Database Creation: The project starts by creating a database named sql_project_p1.
##### Table Creation: A table named google_play_store is created to store the google play store data. The table structure includes columns for app, category, rating, reviews, size, installs, type, price, content rating, genres, last updated, current ver, android ver
```sql
CREATE DATABASE sql_project_p1;

DROP TABLE IF EXISTS google_play_store;
CREATE TABLE google_play_store
			(
				App	VARCHAR(100) PRIMARY KEY,
				Category VARCHAR (30),	
				Rating FLOAT,
				Reviews INT,
				Size VARCHAR(50),	
				Installs VARCHAR(50),	
				Type VARCHAR(50),	
				Price VARCHAR(50),	
				Content_Rating VARCHAR(50),	
				Genres VARCHAR(50),
				Last_Updated DATE,	
				Current_Ver VARCHAR(50),	
				Android_Ver VARCHAR(50)
			);
```

### 2. Data Exploration & Cleaning
##### Record Count: Determine the total number of records in the dataset.
##### Customer Count: Find out how many unique customers are in the dataset.
##### Category Count: Identify all unique product categories in the dataset.
##### Null Value Check: Check for any null values in the dataset and delete records with missing data.
```sql
# Basic Data Exploration
SELECT * from google_play_store LIMIT 10;

SELECT 
	COUNT(*) FROM google_play_store

UPDATE google_play_store
SET rating = NULL
WHERE rating IS NULL OR rating = 'NaN';
```
```sql
# Data Cleaning
SELECT * from google_play_store
WHERE 
	app IS NULL
	OR
	category IS NULL
	OR
	rating IS NULL
	OR
	reviews IS NULL
	OR
	size IS NULL
	OR
	installs IS NULL
	OR
	type IS NULL
	OR
	price IS NULL
	OR
	content_rating IS NULL
	OR
	genres IS NULL
	OR
	last_updated IS NULL
	OR
	current_ver IS NULL
	OR
	android_ver IS NULL
```
```sql
DELETE from google_play_store
WHERE 
	app IS NULL
	OR
	category IS NULL
	OR
	rating IS NULL
	OR
	reviews IS NULL
	OR
	size IS NULL
	OR
	installs IS NULL
	OR
	type IS NULL
	OR
	price IS NULL
	OR
	content_rating IS NULL
	OR
	genres IS NULL
	OR
	last_updated IS NULL
	OR
	current_ver IS NULL
	OR
	android_ver IS NULL
```

### 3. Exploratory Data Analysis (EDA)
```sql
#Counting the number of apps in each category
SELECT category, COUNT(app) AS num_apps
FROM google_play_store
GROUP BY category
ORDER BY num_apps DESC;
```

```sql
#Calculating the average rating for each category
SELECT category, AVG(rating) AS avg_rating
FROM google_play_store
GROUP BY category
ORDER BY avg_rating DESC;
```

```sql
#Summing the installs for each genre
UPDATE google_play_store
SET installs = REPLACE(installs, ',', '');  -- Remove commas
```
```sql
UPDATE google_play_store
SET installs = REPLACE(installs, '+', '');  -- Remove plus sign
```

```sql
ALTER TABLE google_play_store
ALTER COLUMN installs TYPE BIGINT USING installs::BIGINT;  -- Convert to BIGINT
```

```sql
SELECT genres, SUM(CAST(installs AS BIGINT)) AS
total_installs
FROM google_play_store
GROUP BY genres
ORDER BY total_installs DESC
LIMIT 10;
```

```sql
# Counting the number of free and paid apps
SELECT type, COUNT(app) AS num_apps
FROM google_play_store
GROUP BY type;
```

```sql
# Calculating the average price of paid apps
UPDATE google_play_store
SET price = TRIM(BOTH ' ' FROM REPLACE(price, '$', ''));
```

```sql
ALTER TABLE google_play_store
ALTER COLUMN price TYPE NUMERIC USING price::NUMERIC;
```

```sql
SELECT AVG(price) AS avg_price
FROM google_play_store
WHERE type = 'Paid';
```

```sql
# Counting the number of apps for each content rating
SELECT content_rating, COUNT(app) AS num_apps
FROM google_play_store
GROUP BY content_rating;
```

```sql
# Finding the correlation between the number of reviews and rating
SELECT ROUND(CAST(CORR(reviews, rating) AS NUMERIC), 2) AS rounded_correlation
FROM google_play_store;
```

```sql
# Listing the top 10 most expensive apps
SELECT app, price, rating
FROM google_play_store
WHERE type = 'Paid'
ORDER BY price DESC
LIMIT 10;
```

```sql
# Listing the top 10 apps with the highest installs
SELECT app, installs, rating
FROM google_play_store
ORDER BY CAST(installs AS BIGINT) DESC
LIMIT 10;
```

```sql
End of the Project
```

## 6. Conclusion

##### Data Insights: We explored the distribution of app ratings, prices, and installs.
##### We identified the top categories and genres in terms of the number of apps and installs.
##### Machine Learning: We built a Random Forest model to predict app ratings based on the dataset, achieving a decent R-squared score.
##### This project serves as a comprehensive introduction to SQL for data analysts, covering database setup, data cleaning, exploratory data analysis, and business-driven SQL queries. The findings from this project can help drive business decisions by understanding rating system, reviews of apps, and finance reviews.
