# AQI-prediction
🌫️ Air Quality (PM 2.5) Prediction Model
A machine learning project to forecast PM 2.5 concentration using meteorological data and a Linear Regression model.

🎯 Project Goal
The primary objective is to build a predictive model that accurately estimates the Air Quality Index (AQI), specifically the PM 2.5 levels, based on various atmospheric conditions. This helps in understanding the relationship between weather patterns and air pollution.

📊 The Data
The model is trained on a dataset (AQI_Data.csv) containing daily meteorological readings.

Features Used for Prediction:

Index | Feature | Description                | Unit
-----------------------------------------------------
1     | T       | Average Temperature        | °C
2     | TM      | Maximum Temperature        | °C
3     | Tm      | Minimum Temperature        | °C
4     | SLP     | Sea Level Pressure         | hPa
5     | H       | Average Relative Humidity  | %
6     | VV      | Average Visibility         | km
7     | V       | Average Wind Speed         | km/h
8     | VM      | Maximum Wind Speed         | km/h


Target Variable:

PM 2.5: Particulate Matter 2.5 Concentration (µg/m³)

🤖 The Model: Linear Regression
A Linear Regression model from the Scikit-learn library was chosen for this task. The process involves:

Data Cleaning: Removing any rows with missing values to ensure data quality.

Data Splitting: Partitioning the dataset into a 75% training set and a 25% testing set.

Model Training: Fitting the linear regression model on the training data.

Prediction: Using the trained model to make predictions on the unseen test data.

📈 Performance & Results
The model's performance was evaluated on the test set to determine its accuracy and predictive power.

R² Score: 0.502
This means the model can explain approximately 50.2% of the variance in the PM 2.5 data.

Mean Squared Error: 3098.06
This metric gives a quantitative measure of the model's prediction error.

🚀 How to Run
The entire workflow, from data loading to model evaluation, is contained within the AQI(sahil yadav).ipynb Jupyter Notebook. Simply open and run the notebook cells to reproduce the results.

Author
Sahil Yadav
