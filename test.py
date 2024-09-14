from main import LinearRegression, normalize, train_test_split
import numpy as np

x = np.array([[1,2], [2, 3], [3, 4], [4, 5], [5, 6]])
y = np.array([2, 3, 4, 5, 6])

x_normalize = normalize(x)

x_train, x_test, y_train, y_test = train_test_split(x_normalize, y, test_size=0.2)

model = LinearRegression(learning_rate=0.01, n_interations=1000)
model.fit(x_train, y_train)

predictions = model.predict(x_test)
print("Predictions: ", predictions)