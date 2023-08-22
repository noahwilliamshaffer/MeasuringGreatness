# imports
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from tabulate import tabulate

from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import ElasticNet
from sklearn.linear_model import SGDRegressor
from sklearn.ensemble import RandomForestRegressor

plt.style.use('seaborn')
# Reads mvp_stats using pandas and inputs into dataframe
mvp_stats = pd.read_csv("player_mvp_stats.csv")
# Clean data table of values
mvp_stats = mvp_stats.fillna(0)
del mvp_stats["Unnamed: 0"]
mvp_stats["Player"] = mvp_stats["Player"].str.replace("*", "", regex=False)
# Initialise Figure_num
figure_num = 1

# No strings, only numbers
# Cannot use 'Pts Won', 'Pts Max', 'Share' because share is what we are trying to predict
# pts won and pts max are too closly cooralated with share, share = pts won / pts max
predictors = ['Age', 'G', 'GS', 'MP', 'FG', 'FGA', 'FG%', '3P',
              '3PA', '3P%', '2P', '2PA', '2P%', 'eFG%', 'FT', 'FTA', 'FT%', 'ORB',
              'DRB', 'TRB', 'AST', 'STL', 'BLK', 'TOV', 'PF', 'PTS', 'Year', 'W', 'L', 'W/L%', 'GB', 'PS/G',
              'PA/G', 'SRS']


# Compares actual Top 5 share vs predicted Top 5 share and checks the Accuracy
def accuracy(sorted_comp):
    # Sorts by actual share values, output top 5
    actual = sorted_comp.sort_values("Share", ascending=False).head(5)
    # Sorts by predicted share values
    predictions = sorted_comp.sort_values("predictions", ascending=False)
    checker = []
    found = 0
    seen = 1
    # Checks if top 5 actual is contained in top 5 predicted for each one that is not in predicted,
    # it lowers accuracy by 1/5
    for index, row in predictions.iterrows():
        if row["Player"] in actual["Player"].values:
            found += 1
            checker.append(found / seen)
        seen += 1

    top5 = sum(checker) / len(checker)
    return top5


def accuracy_mvp(sorted_comp):
    # Compares actual MVP vs predicted MVP and checks if it is correct
    actual = sorted_comp.sort_values("Share", ascending=False).head(1)
    predictions = sorted_comp.sort_values("predictions", ascending=False).head(1)

    if actual.iloc[0]['Player'] == predictions.iloc[0]['Player']:
        return 1
    return 0


def compound_years_test(model, year):
    # Initilizes Lists used
    average_accuracy = []
    compound_prediction = []
    average_accuracy_mvp = []
    years_list = []
    for year in year:
        # Creates training and testing sets, delimits by year
        # Starts with first 5 years as testing values, and as the year increases,
        # It will add the year it tested on into the training data, at the end it will be
        # Training on 29 years (mvp_stats["Year"] < year) and tests on 1 year (mvp_stats["Year"] == 2021)
        training = mvp_stats[mvp_stats["Year"] < year]
        testing = mvp_stats[mvp_stats["Year"] == year]
        # Fits model, and creates predictions like accuracy()
        model.fit(training[predictors], training["Share"])
        predictions = model.predict(testing[predictors])
        predictions = pd.DataFrame(predictions, columns=["predictions"], index=testing.index)
        # Compares player and share of testing df to predictions df
        compare = pd.concat([testing[["Player", "Share"]], predictions], axis=1)
        # Appends to list to store values
        compound_prediction.append(compare)
        # List of years
        years_list.append(year)
        # List of accuracy of the compared values for top5
        average_accuracy.append(accuracy(compare))
        # List of accuracy of the compared values for mvp
        average_accuracy_mvp.append(accuracy_mvp(compare))
        # Finds total mvp % of all years
        mvp_compound = (sum(average_accuracy_mvp) / len(average_accuracy_mvp)) * 100
        # Finds total top 5 % of all years
        top5_compound = (sum(average_accuracy) / len(average_accuracy)) * 100
        # outputs all top 5 % year by year for graphs into list with correct formatting
        average_accuracy = [float(('%.3f' % float(elem)).strip("0")) for elem in average_accuracy]
        # model name formatting
        str_model = str(model)
        sep = str_model.split("(", 1)
        str_model = sep[0]
        # Running status shows that values for each year, runs in real time
        print("Running " + str_model + " for year " + str(year) + ", Top 5 Accuracy = "
              + '%.3f' % (accuracy(compare) * 100) + "%"
              + ", MVP Accuracy = " + str(accuracy_mvp(compare) * 100) + "%")
    # Cleans model name
    str_model = str(model)
    sep = str_model.split("(", 1)
    str_model = sep[0]
    # keeps track of plt figures
    global figure_num
    # Creates top 5 % graphs for each model
    plt.figure(figure_num)
    plt.scatter(years[5:], average_accuracy, c='blue', edgecolors='black', linewidths=1, alpha=0.7)
    plt.xlabel("Years")
    plt.ylabel("Accuracy of Top 5 MVP Prediction")
    plt.title(str_model + ": Top 5 MVP Prediction")
    plt.savefig(("Results/" + str_model + "_TOP5.png"))

    # Increments figure for graphs for each model
    figure_num = figure_num + 1

    # Creates mvp Graphs for each model
    plt.figure(figure_num)
    plt.scatter(years[5:], average_accuracy_mvp, c='red', edgecolors='black', linewidths=1, alpha=0.7)
    plt.xlabel("Years")
    plt.ylabel("Accuracy MVP Prediction")
    plt.title(str_model + ": MVP Prediction")
    plt.savefig(("Results/" + str_model + "_MVP.png"))

    # Increments figure for graphs for each model

    figure_num = figure_num + 1

    # Returns compounded values as a string
    return '%.3f' % top5_compound + "%        |    " + '%.3f' % mvp_compound + "%"


if __name__ == '__main__':
    # Definition for Linear Models
    ridge = Ridge(alpha=.1)
    las = Lasso(alpha=.001)
    lin = LinearRegression()
    elas = ElasticNet(alpha=.1)
    sgd = SGDRegressor()
    # Definition for ensemble model
    rf = RandomForestRegressor(n_estimators=50, random_state=1, min_samples_split=5)

    years = list(range(1991, 2022))
    # First 5 years are training first test set (1996), next is 6 years for 7th test set, then 7, ect.
    # Creates Dataframe for output table and compound Years test func to display compounded results
    data = {'Model': ['Ridge', 'Lasso', 'Linear', 'Elastic Net', 'SGD', 'Random Forest'],
            'Top 5 Accuracy | MVP Accuracy': [y := compound_years_test(ridge, years[5:]),
                                              h := compound_years_test(las, years[5:]),
                                              compound_years_test(lin, years[5:]),
                                              compound_years_test(elas, years[5:]),
                                              compound_years_test(sgd, years[5:]),
                                              compound_years_test(rf, years[5:])],
            }

    # prints dataframe at the end showing compounded results of each model
    df = pd.DataFrame(data)
    print((tabulate(df, headers='keys', tablefmt='psql')))
    # Makes values from df into lists
    model_list = df['Model'].tolist()
    values_list = df['Top 5 Accuracy | MVP Accuracy'].tolist()
    new_val = []
    for values_list in values_list:
        values_list = values_list.replace(" ", "")
        values_list = values_list.replace("%", "")
        new_val.append(values_list.split('|'))

    # Send values to respective list
    top5_results = []
    mvp_results = []
    for temp in new_val:
        top5_results.append(float(temp[0]))
        mvp_results.append(float(temp[1]))

    # Create bar graph including all model's top 5 MVP and MVP predictions
    model_names = ['Ridge', 'Lasso', 'Linear', 'Elastic Net', 'SGD', 'Random Forest']
    x_axis = np.arange(len(model_names))
    plt.figure(figure_num)
    plt.bar(x_axis + 0.2, top5_results, 0.4, label='Top 5 MVP ')
    plt.bar(x_axis - 0.2, mvp_results, 0.4, label='MVP')
    plt.xticks(x_axis, model_names)
    plt.xlabel("Models")
    plt.ylabel("Accuracy of Prediction (%) ")
    plt.title("Model Predictions")
    plt.legend()
    plt.savefig("Results/MVPBarGraph.png")

    # Also creates graphs as popups
    plt.show()
    plt.close('all')
