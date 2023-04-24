# Guide to Springboard Capstones, Case Studies, and Tutorials

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

## 3. [Predicting the Price of a Property](https://github.com/GHASS19/Predicting-the-Price-of-a-Property)

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


# Case Studies

## 1. [Bayesian Optimization with LightGBM](https://github.com/GHASS19/Bayesian_Optimization_LightGBM_Case_Study)

### Main Ideas
- Predict if a flight departure is going to be delayed by 15 minutes based on the variables and then find the best results.
- Learn how Bayesian Optimization works with a graph of the Gaussian process.
- Test the Bayesian optimization on real flight departures data using the Light GBM.

## 2. [Red Wine with Linear Regression](https://github.com/GHASS19/Linear-Regression-Case-Study-of-the-Red-Wine-Dataset)

### Main Ideas
-  I used linear regression to predict the fixed acidity of red wine using just one variable and then mulitple variables.
- Load and Source the red wine data.
- Exploratory Data Analysis. Displaying heatmaps, pairplot and a few scatterplots.
- Linear Regression Modeling. Our best model was 4. It had an R2 score of .742 and used fewer predictors. 

## 3. [Cowboy Cigarettes Time Series with ARIMA](https://github.com/GHASS19/Cowboy_Cigarettes_Time_Series_Case_Study)

### Main Ideas
- Use the 1949-1960 data to predict the manufacturer's cigarette sales after they stopped in 1960.
- Sourcing and loading the cigarette data. Cleaning, transforming and visualizing or dataset.
- I made the data stationary to prepare it for the ARIMA model. 
- The best p,d, q parameters for our ARIMA model were 2, 1, 1.
- The ARIMA model predicted cigarette sales starting in December of 1960. 
- I concluded that people purchased more cigarettes during the summer possibly due to the good weather, disposable income and time off.

## 4. [Diabetes Grid Search with K-Nearest Neighbor Model](https://github.com/GHASS19/Grid-Search-in-KNN-Model-Case-Study)

### Main Ideas
- Utilized KNN with 31 different neighbors in predicting if a Pima Indian had diabetes or not. 
- The KNN model had an accuracy score of .752644. This was better than the random forest model used as well. 
- This was a Classification problem in which I used cross validation, precision, recall and f1-score to measure model preformance. 

## 5. [Customer Segmentation with K-Means Clustering](https://github.com/GHASS19/K-Means_Clustering_Customer_Segmentation_Case_Study)

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

## 9. [Covid-19 Random Forest Classification Model](https://github.com/GHASS19/Random_Forest_Covid19_Case_Study)

### Main Ideas
- I will predict if a patient is released from the hospital, isolated or deceased from the virus using a dataset from South Korea in Decemberand January of 2020 using a random forest multiclass model.
- Here I had an imbalanced data set. Many of the patients are in isolation compared to the other two states of the patient's health.
- The random forest had an Accuracy of 0.865 and a f1-score of 0.832. The model was very good at predicting if a patient was released at .99.

## 10. [RR Diner Coffee Random Forest Classification Model](https://github.com/GHASS19/RR-Diner-Coffee-Decision-Tree-Case-Study)

### Main Ideas
- I built a decision tree to predict how many units of the Hidden Farm Chinese coffee will be purchased by RR Diner Coffee's most loyal customers. 
- If we predict more than 70% of customers would then we will go ahead with the business contract with Hidden Farm.
- Used scikitlearn to build four different decision tree models — two using entropy and two using gini impurity.
- Then I used a random forest model which predicted that 82.89% of customers would pruchase the coffee and thus we should sign the contract with Hidden Farm.

## 11. [Country Club SQL](https://github.com/GHASS19/SQL-Country-Club-Data)

### Main Ideas
- I answered 13 questions using SQL for a Country Club database.
- Answered questions like, Produce a list of facilities with a total revenue less than 1000.
- Also learned how to use SQL queries in python.

## 12. [London Housing, Looking at Data with Python](https://github.com/GHASS19/London-Housing-Case-Study)

### Main Ideas
- Using the Data Science pipeline, I found which borough of London has seen the greatest average increase in housing prices over the two decades covered by the dataset.
- I created a function that calculated a ratio of house prices, comparing the price of a home in 2018 to the price in 1998.
- Hackney borough had the greatest increase in average home prices at 6.198%.
- The biggest issue was cleaning the data to make sense of what we were working with. The excel format needed to be changed, and some of the values were irrelevant to the majority of the data.
- The biggest questions going forward was, which top 5 London Boroughs had the least amount of increase in average home prices over the 20 year time span. Also what Borough had the biggest fluctuation in prices during that time.

## 13. [API Mini Project with a JSON File](https://github.com/GHASS19/API-Mini-Project-JSON-File)

### Main Ideas
- Using equities data from the Frankfurt Stock Exhange (FSE) using an API key I analyzed the stock prices of a company called Carl Zeiss Meditec.
- I converted the JSON file on the company's equities data into python dictionary.
- Then I answered six questions regarding the data, like what was the largest change in any one day (based on High and Low price)? Which was 2.81 in any one day for 2017.


## 14. [Integrating Phone Apps with a Permutation Test](https://github.com/GHASS19/Integrating-Apps)

## Main Ideas
- The main objective was to find out if Apple Store apps received better reviews than the Google Play apps using a permutation test.
- The Null hypothesis was: the observed difference in the mean rating of Apple Store and Google Play apps is due to chance (and thus not due to the platform).
- The Alternative hypothesis was: the observed difference in the average ratings of apple and google users is not due to chance (and is actually due to platform).
- The observed data is statistically significant. The p-value is 0 and we will reject the null.
- I decided that the platform does have an impact on ratings. I will advise our client to integrate only Google Play into their operating system interface.


## 15. [Frequentist Inference with a T-Test](https://github.com/GHASS19/Frequentist-Inference-A-and-B-Case-Study)

## Main Ideas
- I Sampled and calculating probabilities from a normal distribution. Found the correct way to estimate the standard deviation of a population from a sample.
- Learned how to calculate critical values and confidence intervals.
- Used the central limit theorem to help me apply frequentist techniques to answer questions that pertain to very non-normally distributed data from the real world.
- Performed inference using data to answer business questions for a hospital.
- Forming a hypothesis and framing the null and alternative hypotheses and then testing it with a t-test.

## 16. [Predicting Gender with Logistic Regression](https://github.com/GHASS19/Logistic-Regression-Advanced-Case-Study)

## Main Ideas
- In this case study I built a logistic regression model that uses a person's height & weight to predict their gender with 92% accuracy.
- Used a grid search to find the best parameter to predict the gender was with a 'C' of 0.01.
- If the probability is greater than 0.5, it classifies the sample as type '1' (male), otherwise it classifies the sample to be class '0' with logistic regression.

## 17. [Take Home Challenge with Random Forest, Logistic Regression & Null Hypothesis](https://github.com/GHASS19/Take-Home-Challenge-Ultimate-Technologies-Inc.-)

## Main Ideas
- This case study was to help me prepare for a take home test for an interview with a company.
- The first part was exploratory data analysis of the data.
- The second part was to determine if an experiment was a success. Two cities Gotham and Ultimate Metropolis separated by a bridge have proposed an experiment to encourage public transportation drivers to be available in both cities, by reimbursing all toll costs. They want to know if it is profitable for both cities to do it.
- Using a null hypothesis test on three important variables, (wait times, profit, and bridge toll payments) would determine if the experiment is a success.
- The third part of the take home challenge was to determine if a rider was retained and used their transportation in the preceding 30 days using random forest and logistic regression.
- The cross validation of random forest had the highest accuracy score of .84 at predicting the retention of a rider. The random forest model's precision, recall, auc and cross validation scores were all higher than logistic regression.

## 18. [Take Home Challenge with Random Forest Classifier](https://github.com/GHASS19/Take-Home-Challenge-Relax-Inc-)

## Main Ideas
- Another case study to prepare me for a take home test for an interview. This was about a company called Relax Inc. that wanted to understand their customer retention and why some never came back to use their product.
- The random forest classifier model had an accuracy score of 0.705 and a precision score of 0.154. No the best at predicting their customers habits with the dataset.
- The most important feature in the random forest classifier by far was invited_by_user_id at 0.971558.


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

## V. [Monalco Mining Problem Statement](https://github.com/GHASS19/Monalco_Mining_SMART_problem_statement)

### Main Objectives

- Learned to write a problem statement about a minning company that needs to reduce their maintance cost due to the decrease in the price of iron ore.
- The problem statement includes:
- Context
- Critieria for Success
- Scope of Solution Space
- Constraints within Solution Space
- Stakeholders to Provide key insight
- Key Data Sources
