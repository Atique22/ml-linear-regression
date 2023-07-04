import pandas as pd
import matplotlib as plt
#  load dataset 
data = pd.read_csv("house_data.csv")

print(data.head())


# Create a scatter plot
plt.scatter(data['Size'], data['Price'])
plt.xlabel('Size (in square feet)')
plt.ylabel('Price (in dollars)')
plt.title('House Prices')
plt.show()