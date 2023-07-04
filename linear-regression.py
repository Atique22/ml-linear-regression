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


# User inputs for new property attributes
bedrooms = float(input("Enter the number of bedrooms: "))
bathrooms = float(input("Enter the number of bathrooms: "))
sqft_living = float(input("Enter the square footage of the living area: "))
sqft_lot = float(input("Enter the square footage of the lot: "))
floors = float(input("Enter the number of floors: "))
waterfront = float(input("Enter 1 if the property has a waterfront, 0 otherwise: "))
view = float(input("Enter the view rating (0-4): "))
condition = float(input("Enter the condition rating (1-5): "))
grade = float(input("Enter the grade rating (1-13): "))
sqft_above = float(input("Enter the square footage of the above-ground area: "))
sqft_basement = float(input("Enter the square footage of the basement: "))
yr_built = float(input("Enter the year the property was built: "))
yr_renovated = float(input("Enter the year the property was last renovated, or 0 if never: "))
zipcode = float(input("Enter the property's zipcode: "))
lat = float(input("Enter the latitude of the property: "))
long = float(input("Enter the longitude of the property: "))
sqft_living15 = float(input("Enter the square footage of the interior living space for the nearest 15 neighbors: "))
sqft_lot15 = float(input("Enter the square footage of the land lots of the nearest 15 neighbors: "))

# Predict the price for the new property
new_property = [[bedrooms, bathrooms, sqft_living, sqft_lot, floors, waterfront, view, condition, grade, sqft_above, sqft_basement, yr_built, yr_renovated, zipcode, lat, long, sqft_living15, sqft_lot15]]
predicted_price = model.predict(new_property)

print("Predicted Price: $", predicted_price)

# Generate predictions
y_pred = model.predict(X)

# Plot the data points
plt.scatter(y, y_pred)
plt.xlabel('Actual Price')
plt.ylabel('Predicted Price')

# Plot the best fit line
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--')
plt.show()