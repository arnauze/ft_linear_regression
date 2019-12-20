import sys
import matplotlib.pyplot as plt

# ----------------------------------------------------------
# Utility functions
# ----------------------------------------------------------

def mean(array):
    i = 0
    total = 0
    for i in range(len(array)):
        total += array[i]
    return total / len(array)

def average(array):
    total = 0
    for number in array:
        total += number
    return total / len(array)

def scale(min_x, max_x, X):
    new_X = []
    for i in range(0, len(X)):
        new_X.append((X[i] - min_x) / (max_x - min_x))
    return new_X

# ----------------------------------------------------------
# Other algorithm 
# ----------------------------------------------------------

def ordinary_least_squares(lhs, rhs):
    m = 0
    c = 0
    average_lhs = average(lhs)
    average_rhs = average(rhs)
    x_y = 0
    x_x2 = 0
    for j in range(len(lhs)):
        x_x = lhs[j] - average_lhs
        y_y = rhs[j] - average_rhs
        x_y += x_x * y_y
        x_x2 += (lhs[j] - average_lhs) * (lhs[j] - average_lhs)
    m = x_y / x_x2
    c = average_rhs - m * average_lhs
    return c, m

# ----------------------------------------------------------
# Parsing of the csv file
# ----------------------------------------------------------

def parseLines(lines):
    x = 0
    for line in lines:
        line = line.replace("\n", "")
        split = line.split(',')
        if x == 0:
            X = { "name": split[0], "elements": [] }
            Y = { "name": split[1], "elements": [] }
        else:
            try:
                X["elements"].append(float(split[0]))
                Y["elements"].append(float(split[1]))
            except:
                print("Error in file line " + (x + 1))
        x += 1
    return X, Y

# ----------------------------------------------------------
# Gradient descent algorithm
# ----------------------------------------------------------

def gradient_step(b_current, a_current, X, Y):
    learning_rate = 0.1
    b_gradient = 0
    a_gradient = 0
    N = float(len(X))
    for i in range(0, len(X)):
        x = X[i]
        y = Y[i]
        b_gradient += ((a_current * x + b_current) - y)
        a_gradient += (x * ((a_current * x + b_current) - y))
    new_b = b_current - b_gradient * (learning_rate / N)
    new_a = a_current - a_gradient * (learning_rate / N)
    return new_b, new_a

def gradient_descent(X, Y):
    b = 0
    a = 0
    epochs = 2000
    f = open("Logs", "w+")
    f.write("")
    f.close()
    f = open("Logs", "a")
    for _ in range(epochs):
        b, a = gradient_step(b, a, X, Y)
        f.write(str(getMeanSquaredError(a, b, X, Y)) + "\n")
    f.close()
    return b, a

# ----------------------------------------------------------
# Mean Squared Error (MSE)
# ----------------------------------------------------------

def getMeanSquaredError(a, b, X, Y):
    total = 0
    for i in range(len(X)):
        total += (Y[i] - (a * X[i] + b)) ** 2
    return total / len(X)

# ----------------------------------------------------------
# Matplotlib functions
# ----------------------------------------------------------

# Function outputing the dataset before we use our gradient descent
def showEmptyDataset(X, Y):
    plt.scatter(X["elements"], Y["elements"]) 
    plt.xlabel(X["name"]) 
    plt.ylabel(Y["name"]) 
    plt.title("Data.csv")
    plt.show()

# Function outputting the graph with the given dataset and our line created using our new theta0 and theta1
def showLinearRegression(X, Y, theta0, theta1):
    new_Y = []
    for x in X:
        new_Y.append(theta1 * x + theta0)
    plt.scatter(X, Y) 
    plt.plot(X, new_Y)
    plt.show()

# ----------------------------------------------------------
# Training
# ----------------------------------------------------------

def train():
    if len(sys.argv) < 2:
        print("usage: python3 training.py <dataset>")
    else:
        f = open(sys.argv[1], "r")
        lines = f.readlines()
        f.close()
        X, Y = parseLines(lines)
        if (len(sys.argv) > 2 and ("-v" in sys.argv[2] or "-ov" in sys.argv[2])):
            showEmptyDataset(X, Y)
        if (len(sys.argv) > 2 and ("-o" in sys.argv[2] or "-ov" in sys.argv[2])):
            theta0, theta1 = ordinary_least_squares(X["elements"], Y["elements"])    
        else:
            min_x = min(X["elements"])
            max_x = max(X["elements"])
            scaled_X = scale(min_x, max_x, X["elements"])
            theta0, theta1 = gradient_descent(scaled_X, Y["elements"])
            theta1 = (theta1 / ( max_x - min_x))
        if (len(sys.argv) > 2 and ("-v" in sys.argv[2] or "-ov" in sys.argv[2])):
            showLinearRegression(X["elements"], Y["elements"], theta0, theta1)
        f = open("values.py", "w")
        s = "theta0=" + str(theta0) + "\ntheta1=" + str(theta1)
        f.write(s)
        f.close()
    return


if __name__ == "__main__":
    train()