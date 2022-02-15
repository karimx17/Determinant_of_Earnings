import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# IF YOU ARE A RECRUITER AND ARE NOT FAMILIAR WITH PYTHON, SHOW THIS TO A SOFTWARE DEVELOPER AND THEY CAN RUN IT
# Once you run, 3 graphs will open up

df_data = pd.read_csv('NLSY97_subset.csv')

# Remove duplicates
df_data.drop_duplicates(inplace=True)


# A seaborn displot, that will be used for three graphs
def displot(x, xtext, title):
    with sns.axes_style("darkgrid"):
        ax = sns.displot(
            data=df_data,
            x=x,
            kde=True,
        )
        ax.set(
            ylabel="Count",
            xlabel=xtext,
            title=title
        )

    plt.show()


# The amount of Schooling the US population has
displot("S", 'Years of Schooling', "2011 Years of Schooling")

# The amount of Experience the US population has
displot("EXP", 'Years of Experience', "2011 Years of Experience")

# The amount of Earnings/hour the US population has
displot("EARNINGS", 'Earnings per Hour', "2011 Earnings")

regression = LinearRegression()


# Simple Linear Regression
def simple_regression():

    # Split Training and Test Data Set
    X = pd.DataFrame(df_data, columns=["S"])
    y = pd.DataFrame(df_data, columns=['EARNINGS'])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=10)

    regression.fit(X_train, y_train)
    r_sqaured = round(regression.score(X_train, y_train), 2) * 100

    print(f"Linear: Training data r-squared: {r_sqaured} %")


simple_regression()


# Multivariable Regression
def multivariable_regression():
    X2 = pd.DataFrame(df_data, columns=["S", "EXP"])
    y2 = pd.DataFrame(df_data, columns=['EARNINGS'])

    # Split Training and Test Data Set
    X_train2, X_test2, y_train2, y_test2 = train_test_split(X2, y2, test_size=0.2, random_state=10)

    regression.fit(X_train2, y_train2)
    r_sqaured = round(regression.score(X_train2, y_train2), 2) * 100

    print(f"Multivariable: Training data r-squared: {r_sqaured} %")


multivariable_regression()


def make_prediction(schooling_years, amount_experience):
    features = df_data[["S", "EXP"]]
    avg_values = features.mean().values

    df_predict = pd.DataFrame(data=avg_values.reshape(1, len(features.columns)), columns=features.columns)

    # Predictions
    df_predict["S"] = schooling_years
    df_predict["EXP"] = amount_experience
    earnings_list = regression.predict(df_predict)[0]

    # Making output look pretty
    earnings = ", ".join( repr(e) for e in earnings_list)
    earnings = round(float(earnings), 2)

    if earnings > 0:
        print(f"Based on {schooling_years} years of school and {amount_experience}"
              f" years of experience you will make: ${earnings}/hr")
    else:
        print("Not enough schooling or experience to conduct prediction")


# Input years of schooling and work experience
make_prediction(16, 16)
