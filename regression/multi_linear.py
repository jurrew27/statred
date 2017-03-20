import numpy as np
import csv


def read_csv():
    with open('housesRegr.csv', newline='\r') as f:
        reader = csv.reader(f)
        headers = next(reader)[0].split(';')
    data = np.loadtxt('housesRegr.csv', dtype=int, delimiter=';', skiprows=1)
    return headers, data


def h_func(thetas, predictors):
    result = 0
    for i in range(len(thetas)):
        result += thetas[i] * predictors[i, :]
    return result


def regress(predictors, target, a):
    m, n = predictors.shape

    x0 = np.ones(m)
    predictors = np.insert(predictors, 0, x0, axis=1)
    m, n = predictors.shape

    thetas = np.zeros(n).transpose()

    thetaChecks = np.ones(n)
    divisor = a / m
    corrects = 0

    i = 0
    while corrects != n:
        h = np.sum(h_func(thetas, predictors))
        corrects = 0

        for i in range(n):
            if abs(thetas[i] - thetaChecks[i]) > 10 ** -5:
                thetaChecks[i] = thetas[i]
                thetas[i] -= divisor * np.sum((h - target) * predictors[:, i])
            else:
                corrects += 1
        i += 1
    return i, thetas


if __name__ == '__main__':
    headers, data = read_csv()
    X = data[:, 1:4]
    Y = data[:, 4]

    i, thetas = regress(X, Y, 10 ** -10)

    m, n = X.shape
    x0 = np.ones(m)
    X = np.insert(X, 0, x0, axis=1)

    hypoHouseprizes = h_func(thetas, X.transpose()).astype(int)
    print("Standard deviation is %.2d :(" % np.std(abs(Y - hypoHouseprizes)))
