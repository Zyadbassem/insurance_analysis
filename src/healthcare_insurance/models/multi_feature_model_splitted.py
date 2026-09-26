from sklearn.linear_model import LinearRegression
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


def multi_feature_model_splitted(X, Y, Xtest, Ytest):
    lm = LinearRegression()
    lm.fit(X, Y)
    
    yHat = lm.predict(X=Xtest)

    ax1 = sns.histplot(Ytest, color='b', label='Actual', kde=True)
    sns.histplot(yHat, color='r', label='Predicted', kde=True, ax=ax1)

    plt.legend()
    plt.show()

    return yHat