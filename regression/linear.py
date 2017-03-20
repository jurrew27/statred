import numpy as np
import matplotlib.pyplot as plt
import csv


def read_csv():
    with open('housesRegr.csv', newline='\r') as f:
        reader = csv.reader(f)
        headers = next(reader)[0].split(';')
    data = np.loadtxt('housesRegr.csv', dtype=int, delimiter=';', skiprows=1)
    return headers, data


def h_func(th0, th1, predictor):
    return th0 + (th1 * predictor)


def regress(predictor, target, a):
    th0 = 0
    th1 = 0
    th0_check = -1
    th1_check = -1
    m = len(predictor)
    divisor = a / m

    i = 1
    while True:
        h = h_func(th0, th1, predictor)

        if abs(th0 - th0_check) > 10 ** -7:
            th0_check = th0
            th0 -= divisor * np.sum(h - target)
        if abs(th1 - th1_check) > 10 ** -7:
            th1_check = th1
            th1 -= divisor * np.sum((h - target) * predictor)
        else:
            return i, th0, th1

        i += 1


if __name__ == '__main__':
    headers, data = read_csv()
    modelFields = [('Bedrooms', 'Price', .1), ('Bathrooms', 'Price', .1), ('Size', 'Price', 10 ** -8)]

    nPlots = len(modelFields)
    iPlot = 1
    for predictor, target, a in modelFields:
        X = data[:, headers.index(predictor)]
        Y = data[:, headers.index(target)]

        n, th0, th1 = regress(X, Y, a)
        print('Found regression of %s as a function of %s in %d iterations' % (target, predictor, n))

        plt.subplot(1 + nPlots // 2, 2, iPlot)
        plt.plot(X, Y, 'bo')
        plt.plot(X, h_func(th0, th1, X), 'r-')
        plt.xlabel(predictor)
        plt.ylabel(target)

        iPlot += 1

    plt.show()
