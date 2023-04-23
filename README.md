# Guide to Springboard Capstones, Tutorials and Case Studies

![image](https://user-images.githubusercontent.com/86930309/233186316-14d6a5d3-ac53-4225-896d-691c50f22848.png)

# Capstones

## 1. [Adult Weekend Ski Ticket Price Predictor](https://github.com/GHASS19/Ski_Ticket_Price_Predictor_Capstone)

### Main Ideas

- Test two models to predict the price of a ski ticket for an adult on the weekend at Big Moutain Ski Resort in Montana based upon certain varibles from all ski resorts in America.

- I used cross validation in linear regression and random forest to predict the price.

- The random forest model was more accurate. It had a lower mean absolute error by over $1 then the other model.

### Key Findings

- Fast Quad lifts, the amount of ski runs, snow making and vertical drop in feet were the most important features to predicting the price in both the linear regression and random forrest models.

- Big Mountain offers more facilities and skiable terrain than most other resorts in the United States.

- In both of our models vertical drop, snow making, total chairs, fast quads, runs, longest run, trams and skiable terrain were the most important to guests.

- The resort is in the top portion of fast quads, runs, snow making, total chairs and vertical drop compared to other places.

### Recommendations

- Big Mountain Resort should increase their weekend price to $95.87 from $81.00.

- I would do an experiment for a certain amount of time and close just one ski run on the weekend to reduce maintenance cost.

- Adding one run, 150 vertical feet and a new chairlift, (fast quad) is what I highly suggest the resort enact. This plan justifies a ticket price increase of $1.99 that improves revenue to $3,474,638 per year.

- Big Mountain needs to invest in facilities that guests value the most on their ski trips.

- The resort needs data on maintenance cost to make a solid decision going forward.

## 2. [Quaterback Touchdown Prediction Model](https://github.com/GHASS19/QB_Touchdown_Prediction_Capstone)

### Main Ideas

- I created a random forest model to predict how many TDs a QB will throw in an NFL game based upon variables from every QB in the last 20 years worth of data.

- Linear Regresssion, Ridge Regression and Lasso Regression were the other three models I tried on the data.

- The random forest model had a very solid MAE of .006 and RMSE of .06 on the test data. The CV average & Grid Search CV for Random Forest had a very high R2 score of .9977 and .9979.

### Key Findings

- According to the heat map the top three correlations were Attempts & Completions at .94, Attempts & TDs at .87 and Completions & Yards at .91.

- The highest negitive correlation was Interceptions & Rate at -.44.

- The distribution of the data showed us that many of the touchdowns thrown in a game were in the range of 0-3 and that there was not many 4-7 touchdowns.

### Recomendations
 
 - Run models on every stat we have on defense, special teams and offense to predict how every player might perform at each position. This would help to evaluate free agents and our own players.
 
 - We should also run analytics on situational football. Finding out the probability of converting a first down on a 4th & 2 from our 26th yard line would be beneficial to the coach making a decision to go for it or not.

3. [Predicting the Price of a Property](https://github.com/GHASS19/Predicting-the-Price-of-a-Property)

### Main Ideas

- I tested four different models with a standard scaler on each one for my supervised regression price prediction of a home in the Seattle metro area.

- The most accurate R2 score on the test data was Linear Regression at .4696. It would be hard to predict the price of a home if you get less than half of them correct.

### Key Findings

- There was many outliers with a high amount of square feet for living, above and below ground level.

- The majority of homes that were sold had a good condition rating of 3-5.

- As the number of bedrooms increased so did the amount of bathrooms in a particular house increase too.

- Most of the homes had between three and six bedrooms.

- There was a positive correlation between price and square feet of living space at .43.

### Recomendations

- The company needs to obtain or collect more data to create a better regression model to predict the price of a home for their buyers and sellers.The more information we can obtain the better we can help serve our potential clients. This would give us a competitive advantage over other real estate firms.

- I also suggest running models on all the variables to help clients. This would help them understand how much square feet of living space you can expect with the other independent variables. Or how many bedrooms they could expect with certain X variables.

# Springboard Tutorials

## I. [Intro to Gradient Boosting](https://github.com/GHASS19/Intro_to_Gradient_Boosting)

### Main Objectives

 - Understand the conceptual difference between bagging and boosting ensembles.

- Understand how gradient boosting works for regression tasks.

- Learn how to tune the key hyperparameters of gradient boosting ensembles.

## II. [Data Cleaning Exercise](https://github.com/GHASS19/Data_Cleaning_Exercise)

 ### Main Objectives
 
- Was to clean the climbing data to prepare it for modeling and machine learning.

- What to do with NaN values.

## III. [Automated Feature Engineering](https://github.com/GHASS19/Feature_Engineering)

### Main Objectives

- Using automated feature engineering as to build hundreds or thousands of relevant features from a relational dataset.

- Calculating a feature matrix with several hundred relevant features for predicting customer churn.

- Ensuring that our features are made with valid data for each cutoff time.

## IV. [Real Estate Investment Plan](https://github.com/GHASS19/Real_Estate_Investment_Plan)

### Main Objectives

- Using a real estate database, find a state for an ivestment property with optimum traits.

- Create graphs in matplotlib to analyze where to purchase the investment property.

- Based upon the data the best place to purchase was the state was Massachusettes. That had a four bed and two bathroom home for less than the mean price.

# Case Studies

## 1. [Bayesian Optimization LightGBM](https://github.com/GHASS19/Bayesian_Optimization_LightGBM_Case_Study)

### Main Ideas

- Predict if a flight departure is going to be delayed by 15 minutes based on the variables and then find the best results.

- Learn how Bayesian Optimization works with a graph of the Gaussian process.

- Test the Bayesian optimization on real flight departures data using the Light GBM.

## 2. [Linear Regression Red Wine Study](https://github.com/GHASS19/Linear-Regression-Case-Study-of-the-Red-Wine-Dataset)

### Main Ideas

-  I used linear regression to predict the fixed acidity of red wine using just one variable and then mulitple variables.

- Load and Source the red wine data.

- Exploratory Data Analysis. Displaying heatmaps, pairplot and a few scatterplots.

- Linear Regression Modeling. Our best model was 4. It had an R2 score of .742 and used fewer predictors. 

## 3. [Cowboy Cigarettes Time Series ARIMA](https://github.com/GHASS19/Cowboy_Cigarettes_Time_Series_Case_Study)

### Main Ideas

- Use the 1949-1960 data to predict the manufacturer's cigarette sales after they stopped in 1960.

- Sourcing and loading the cigarette data. Cleaning, transforming and visualizing or dataset.

- I made the data stationary to prepare it for the ARIMA model. 

- The best p,d, q parameters for our ARIMA model were 2, 1, 1.

- The ARIMA model predicted cigarette sales starting in December of 1960. 

- I concluded that people purchased more cigarettes during the summer possibly due to the good weather, disposable income and time off.

## 4. [Grid Search in K-Nearest Neighbor Model Case Study](https://github.com/GHASS19/Grid-Search-in-KNN-Model-Case-Study)

### Main Ideas

- Utilized KNN with 31 different neighbors in predicting if a Pima Indian had diabetes or not. 

- The KNN model had an accuracy score of .752644. This was better than the random forest model used as well. 

- This was a Classification problem in which I used cross validation, precision, recall and f1-score to measure model preformance. 

## 5. [K-Means Clustering Customer Segmentation](https://github.com/GHASS19/K-Means_Clustering_Customer_Segmentation_Case_Study)

### Main Ideas

- I used K-Means clustering to find out how many groups we could categorize customers in who would potentially purchase wine from the company.

- The data was marketing newsletters, e-mail campaigns and transactions.

- The Principal Components Analysis expresses that the first four to five components explain a majority of the variance.

## 6. [Cosine Similarity](https://github.com/GHASS19/Cosine_Similarity_Case_Study)

### Main Ideas
-  In this case study I used the cosine similarity to compare a numeric data within a plane of (5,5). The I used a scatter plot to view the way the similarity is calculated using the Cosine angle.

- Also used a text dataset for string matching to test the similarity measure between two different documents, (i.e. Document1 = "Starbucks Coffee" and Document2 = "Essence of Coffee"). 

## 7. [Euclidean and Manhattan Distance](https://github.com/GHASS19/Euclidean_and_Manhattan_Distances_Case_Study)

### Main Ideas

- In this case study I learned the difference between euclidean and Manhattan distances using a colorbar in python. The data set had three coordinates of X,Y and Z.

- First I made a scatter plot of all the euclidean distances of Y and Z in our dataset to our selected location of Y=5, Z=5. 

- Then I made a scatter plot of the manhattan distance of each point to our reference point of X=4, Z=4.

## 8. [Titanic Gradient Boosting](https://github.com/GHASS19/Titanic-Gradient-Boosting-Case-Study)

### Main Ideas

- Predict if a passenger survived the from the Titanic crashing into the iceberg in 1912.

- Utilize gradient boosting to improve predictions based on information from the residuals.

- Had a 87% prediction rate from the AUC ROC curve for the gradient boosting model.

- Precision: Did not survive 0 = 83%. Survived 1 = 88%.

- Recall: Did not survive 0 = 75%. Survived 1 = 93%.

- F1-Score: Did not survive 0 = 79%. Survived 1 = 90%.

## 9. [Random Forest Classifier Model Covid-19](https://github.com/GHASS19/Random_Forest_Covid19_Case_Study)

### Main Ideas

- I will predict if a patient is released from the hospital, isolated or deceased from the virus using a dataset from South Korea in Decemberand January of 2020 using a random forest multiclass model.

- Here I had an imbalanced data set. Many of the patients are in isolation compared to the other two states of the patient's health.

- The random forest had an Accuracy of 0.865 and a f1-score of 0.832. The model was very good at predicting if a patient was released at .99.

## 10. [RR Diner Coffee Random Forest Classification Model](https://github.com/GHASS19/RR-Diner-Coffee-Decision-Tree-Case-Study)

### Main Ideas

- I built a decision tree to predict how many units of the Hidden Farm Chinese coffee will be purchased by RR Diner Coffee's most loyal customers. 

- If we predict more than 70% of customers would then we will go ahead with the business contract with Hidden Farm.

- Used scikitlearn to build four different decision tree models — two using entropy and two using gini impurity.

-  Then I used a random forest model which predicted that 82.89% of customers would pruchase the coffee and thus we should sign the contract with Hidden Farm.
