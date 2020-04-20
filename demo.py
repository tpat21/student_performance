import pandas as pd
import numpy as np
import sklearn
from sklearn import linear_model
import matplotlib.pylab as plt



def compute_error(b, m, data):
    error = 0
    length = len(data)
    for i in range(0, length):
        x = data[i, 0]
        y = data[i, 1]
        error = error + (y - (m * x + b)) ** 2
    return error / float(length)



def step_gradient(b_current, m_current, data, learning_rate):
    b_gradient = 0
    m_gradient = 0
    N = float(len(data))
    for i in range(0, len(data)):
        x = data[i, 0]
        y = data[i, 1]
        b_gradient = b_gradient + -(2 / N) * (y - ((m_current * x) + b_current))
        m_gradient = m_gradient + -(2 / N) * x * (y - ((m_current * x) + b_current))
    new_b = b_current - (learning_rate * b_gradient)
    new_m = m_current - (learning_rate * m_gradient)
    return [new_b, new_m]



def gradient_descent_runner(initial_b, initial_m,data, learning_rate, num_iterations):
    b = initial_b
    m = initial_m

    for i in range(num_iterations):
        b, m = step_gradient(b, m, np.array(data), learning_rate)
    return [b, m]



def show_plot(X, y, m, b):
    plt.scatter(X, y)
    x = np.linspace(25, 75, 100)
    linear_line = m * x + b
    plt.plot(x, linear_line, '-r', label='y=mx+b')
    plt.title('Graph of y=mx+b')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend(loc='upper left')
    plt.grid()
    plt.show()



def run():
    data = pd.read_csv("data.csv", sep=",")
    data = data[["hours", "grade"]]
    predict = "grade"

    # Hyperparameters
    learning_rate = 0.0001
    initial_b = 0
    initial_m = 0
    num_iterations = 1000

    X = np.array(data.drop([predict], 1))
    y = np.array(data[predict])

    x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size=.10, random_state=42)

    linear = linear_model.LinearRegression(fit_intercept=True)
    linear.fit(x_train, y_train)

    acc = linear.score(x_test, y_test)
    error = (abs((y - linear.predict(X) ** 2).mean()))

    print("Accuracy: {:.4f}".format(acc))
    print("Error: {:.4f}".format(float(error)))


    # compute_error(initial_b, initial_m, data)

    [b, m] = gradient_descent_runner(initial_b, initial_m,data, learning_rate, num_iterations)

    print("b value: {:.4f}".format(float(b)))
    print("m value: {:.4f}".format(float(m)))

    show_plot(X, y, m, b)



if __name__ == '__main__':
    run()
