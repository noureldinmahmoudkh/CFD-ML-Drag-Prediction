import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor
# Set random seed for reproducibility
np.random.seed(42)
#load dataset
data=pd.read_csv("cfd_dataset.csv")
#Features (input) and target (output)
x=data[["Reynolds_Number"]]
y=data["Drag_Coefficient"]
#Split data (training+testing)
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=42
)
#polynomial transformation
poly=PolynomialFeatures(degree=3)
x_train_poly=poly.fit_transform(x_train)
x_test_poly=poly.transform(x_test)
#Create model
# model=LinearRegression()
# model.fit(x_train_poly,y_train)
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(x_train,y_train)
#train model
# model.fit(x_train,y_train)
#Predictons
y_pred=model.predict(x_test)
#sort for smooth plotting
sorted_indices=x_test["Reynolds_Number"].argsort()
x_sorted=x_test.iloc[sorted_indices]
y_sorted=y_test.iloc[sorted_indices]
y_pred_sorted=y_pred[sorted_indices]
#Evaluation
mse=mean_squared_error(y_test,y_pred)
r2=r2_score(y_test,y_pred)
print("Mean Squared Error: ",mse)
print("R^2 Score: ",r2)
# 1. Generate a highly dense set of inputs (500 points between 10 and 200)
# We put it in a DataFrame so it matches the feature names the model expects
x_dense = pd.DataFrame({"Reynolds_Number": np.linspace(10, 200, 500)})

# 2. Get the model's predictions for all 500 points
y_dense_pred = model.predict(x_dense)

# 3. Plot the real test data dots
plt.scatter(x_test, y_test, color='blue', label="Real Test Data")

# 4. Plot the dense Random Forest prediction
plt.plot(x_dense, y_dense_pred, color='red', label="Random Forest Model")

plt.xlabel("Reynolds Number (Re)")
plt.ylabel("Drag Coefficient (Cd)")
plt.title("Random Forest Regressor (Step-Function)")
plt.legend()
plt.savefig("results/model_plot.png")  
plt.show()                             