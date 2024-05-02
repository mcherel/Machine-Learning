import scatter
# Data eparation into test and train
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics
import numpy as np

X_train, X_test, y_train, y_test = train_test_split(scatter.X, scatter.y, test_size=0.33, random_state=0)
regressor = LinearRegression()
#Training
regressor.fit(X_train, y_train)

theta1 = regressor.coef_
theta0 = regressor.intercept_

print(f'Theta1 = {theta1}')
print(f'Theta0 = {theta0}')

# Adding a red line
ordonne = np.linspace(0,11,1000)#creates a table of 1000 values from 0 to 15
#graph plot
scatter.plt.plot(ordonne,theta1[0]*ordonne + theta0, color='r')

# Prediction from test base
y_predict = regressor.predict(X_test)

# Metrics
print(f'MAE: \t{metrics.mean_absolute_error(y_test, y_predict)}')
print(f'MSE: \t{metrics.mean_squared_error(y_test, y_predict)}')
print(f'RMSE: \t{np.sqrt(metrics.mean_squared_error(y_test, y_predict))}')
print(f'R²: \t{metrics.r2_score(y_test, y_predict)}')


print(f'Predicted salary for 5 years of experience: {regressor.predict(scatter.pd.DataFrame([5], columns=["YearsExperience"]))[0]}')



# show graph after prediction
scatter.plt.show()