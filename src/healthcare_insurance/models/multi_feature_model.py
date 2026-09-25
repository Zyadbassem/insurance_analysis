from sklearn.linear_model import LinearRegression
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


def multi_feature_model(df: pd.DataFrame, features: list):
    X = df[features]
    Y = df['charges']

    lm = LinearRegression()
    lm.fit(X, Y)

    yHat = lm.predict(X=X)

    ax1 = sns.histplot(Y, color='b', label='Actual', kde=True)
    sns.histplot(yHat, color='r', label='Predicted', kde=True, ax=ax1)

    plt.legend()
    plt.show()

    return yHat