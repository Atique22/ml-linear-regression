import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

#  load dataset 
data = pd.read_csv("house_data.csv")

print(data.head())




# Separate the features (X) and target variable (y)
X = data[['bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot', 'floors', 'waterfront', 'view', 'condition', 'grade', 'sqft_above', 'sqft_basement', 'yr_built', 'yr_renovated', 'zipcode', 'lat', 'long', 'sqft_living15', 'sqft_lot15']]
y = data['price']

# Create a linear regression model
model = LinearRegression()

# Fit the model to the data
model.fit(X, y)

# Generate predictions
y_pred = model.predict(X)

# Plot the data points
plt.scatter(y, y_pred)
plt.xlabel('Actual Price')
plt.ylabel('Predicted Price')

# Plot the best fit line
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--')
plt.show()