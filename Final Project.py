import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn import metrics
import warnings

# Suppress warnings
warnings.filterwarnings("ignore")

# Load the dataset
gold_data = pd.read_csv(r'gld_price_data.csv')

# Print first 5 rows in the dataframe
print(gold_data.head())

# Print last 5 rows of the dataframe
print(gold_data.tail())

# Number of rows and columns
print(gold_data.shape)

# Getting some basic information about the data
print(gold_data.info())

# Checking the number of missing values
print(gold_data.isnull().sum())

# Handling missing values (optional)
gold_data = gold_data.dropna()  # Drop rows with missing values

# Getting statistical measures of the data
print(gold_data.describe())

# Correlation heatmap
correlation = gold_data.drop(columns=['Date']).corr()
plt.figure(figsize=(8, 8))
sns.heatmap(correlation, cbar=True, square=True, fmt='.1f', annot=True, annot_kws={'size': 8}, cmap='Blues')
plt.show()

# Correlation values of GLD
print(correlation['GLD'])

# Checking the distribution of the GLD Price
sns.histplot(gold_data['GLD'], color='green', kde=True)
plt.show()

# Prepare the data
X = gold_data.drop(['Date', 'GLD'], axis=1)
Y = gold_data['GLD']

# Print features (X) and target (Y)
print(X)
print(Y)

# Split data into training and test sets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=2)

# Initialize and train the model
regressor = RandomForestRegressor(n_estimators=100)
regressor.fit(X_train, Y_train)

# Predict on the test data
test_data_prediction = regressor.predict(X_test)
print(test_data_prediction)

# Calculate R-squared error
error_score = metrics.r2_score(Y_test, test_data_prediction)
print(f"R squared error: {error_score}")

# Convert Y_test to list for plotting
Y_test = list(Y_test)

# Plot actual vs predicted values
plt.plot(Y_test, color='blue', label='Actual Value')
plt.plot(test_data_prediction, color='green', label='Predicted Value')
plt.title('Actual Price vs Predicted Price')
plt.xlabel('Number of values')
plt.ylabel('GLD Price')
plt.legend()
plt.show()