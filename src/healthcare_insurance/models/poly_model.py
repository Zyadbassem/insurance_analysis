from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


def poly_model(x, y, x_test, y_test, degree=2):
    poly = PolynomialFeatures(degree)
    x_poly = poly.fit_transform(x)

    lm = LinearRegression()
    lm.fit(x_poly, y)

    poly_test = poly.fit_transform(x_test)
    y_hat = lm.predict(poly_test)


    ax1 = sns.histplot(y_test, label='Actual', color='b', kde=True)
    sns.histplot(y_hat, label='Predicted', color='r', kde=True, ax=ax1)


    plt.legend()
    plt.show()

    return y_hat

