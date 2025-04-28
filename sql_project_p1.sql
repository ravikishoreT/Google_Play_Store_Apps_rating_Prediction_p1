-- SQL Google Play Store - p3
CREATE DATABASE sql_project_p1;


--Create Table
DROP TABLE IF EXISTS google_play_store;
CREATE TABLE google_play_store
			(
				App	VARCHAR(100) PRIMARY KEY,
				Category VARCHAR (30),	
				Rating FLOAT,
				Reviews INT,
				Size VARCHAR(10),	
				Installs VARCHAR(15),	
				Type VARCHAR(10),	
				Price VARCHAR(10),	
				Content_Rating VARCHAR(20),	
				Genres VARCHAR(30),
				Last_Updated DATE,	
				Current_Ver VARCHAR(20),	
				Android_Ver VARCHAR(20)
			);
ALTER TABLE google_play_store
ALTER COLUMN Genres TYPE VARCHAR(50);

--Basic Data Exploration
SELECT * from google_play_store LIMIT 10;

SELECT 
	COUNT(*) FROM google_play_store

UPDATE google_play_store
SET rating = NULL
WHERE rating IS NULL OR rating = 'NaN';

-- Data Cleaning
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

--
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

-- Exploratory Data Analysis (EDA)

-- Counting the number of apps in each category

SELECT category, COUNT(app) AS num_apps
FROM google_play_store
GROUP BY category
ORDER BY num_apps DESC;


-- Calculating the average rating for each category

SELECT category, AVG(rating) AS avg_rating
FROM google_play_store
GROUP BY category
ORDER BY avg_rating DESC;


-- Summing the installs for each genre

UPDATE google_play_store
SET installs = REPLACE(installs, ',', '');  -- Remove commas

UPDATE google_play_store
SET installs = REPLACE(installs, '+', '');  -- Remove plus sign

ALTER TABLE google_play_store
ALTER COLUMN installs TYPE BIGINT USING installs::BIGINT;  -- Convert to BIGINT

SELECT genres, SUM(CAST(installs AS BIGINT)) AS
total_installs
FROM google_play_store
GROUP BY genres
ORDER BY total_installs DESC
LIMIT 10;


-- Counting the number of free and paid apps

SELECT type, COUNT(app) AS num_apps
FROM google_play_store
GROUP BY type;


-- Calculating the average price of paid apps

UPDATE google_play_store
SET price = TRIM(BOTH ' ' FROM REPLACE(price, '$', ''));

ALTER TABLE google_play_store
ALTER COLUMN price TYPE NUMERIC USING price::NUMERIC;

SELECT AVG(price) AS avg_price
FROM google_play_store
WHERE type = 'Paid';


-- Counting the number of apps for each content rating

SELECT content_rating, COUNT(app) AS num_apps
FROM google_play_store
GROUP BY content_rating;


-- Finding the correlation between the number of reviews and rating

SELECT ROUND(CAST(CORR(reviews, rating) AS NUMERIC), 2) AS rounded_correlation
FROM google_play_store;


-- Listing the top 10 most expensive apps

SELECT app, price, rating
FROM google_play_store
WHERE type = 'Paid'
ORDER BY price DESC
LIMIT 10;


-- Listing the top 10 apps with the highest installs

SELECT app, installs, rating
FROM google_play_store
ORDER BY CAST(installs AS BIGINT) DESC
LIMIT 10;

-- End of the Project
