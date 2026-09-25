from sklearn.linear_model import LinearRegression
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def simple_model(df: pd.DataFrame, feature: str):
    X = df[[feature]]
    Y = df['charges']

    lm = LinearRegression()

    lm.fit(X, Y)

    yHat = lm.predict(X)

    ax1 = sns.histplot(Y, color='b', label='Actual')
    sns.histplot(yHat, color='r', label='Predicted', ax=ax1)
    plt.legend()
    plt.show()

    return yHat


