import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv('Salary_Data.csv')
#print(data.head())
#print(data.info())

X = data[['YearsExperience']]
y = data['Salary']




# graph
plt.scatter(X,y)
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.title('Salary Prediction')

# show graph befor prediction
# plt.show()

