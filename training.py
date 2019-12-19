import sys
# from estimate import estimatePrice

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

def scale(min_x, max_x, X):
    new_X = []
    for i in range(0, len(X)):
        new_X.append((X[i] - min_x) / (max_x - min_x))
    return new_X

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

def     gradient_descent(X, Y):
    b = 0
    a = 0
    epochs = 2000
    for _ in range(epochs):
        b, a = gradient_step(b, a, X, Y)
    return b, a

def train():
    if len(sys.argv) != 2:
        print("usage: python3 training.py <dataset>")
    else:
        f = open(sys.argv[1], "r")
        lines = f.readlines()
        X, Y = parseLines(lines)
        min_x = min(X["elements"])
        max_x = max(X["elements"])
        scaled_X = scale(min_x, max_x, X["elements"])
        theta0, theta1 = gradient_descent(scaled_X, Y["elements"])
        theta1 = (theta1 / ( max_x - min_x))
        f.close()
        f = open("values.py", "w")
        s = "theta0=" + str(theta0) + "\ntheta1=" + str(theta1)
        f.write(s)
        f.close()
    return


if __name__ == "__main__":
    train()
