#Predicting NBA Season MVP

## Project Description

The pinnacle of basketball performance is measured in two ways, winning the NBA championship and receiving the award for most valuable player. As an individual, the primary criteria for measuring greatness is the most valuable player award, and it is one of the most hotly debated topics over the course of the NBA season. Many factors play into determining who will attain this crowning achievement. In an attempt to remove biases from the discussion, I implemented various machine learning algorithms that are able to accurately predict the MVP. Furthermore, I sought to analyze which of these models appear to be most well suited to this problem type.

#### Machine Learning Models Used:
1) Ridge Regression
2) LASSO Regression
3) Linear Regression
4) Elastic Net Regression
5) Stochatsic Gradient Descent (SGD)
6) Random Forest Regression

## How to Install the Program

#####  ***The program's code is written completely in Python; A Python Environment will be neccessary to run the program.***

Donwload zip file "MeasuringGreatness.zip" from Canvas. Unzip "MeasuringGreatness.zip" within python environment.

#### Included Files in "MeasuringGreatness.zip":
1) DataFiles/mvps.csv

2) DataFiles/players.csv

3) Results/.placeholder

4) main.py

5) player_mvp_stats.csv

## Required Modules
1) Pandas :   _pip install -U pandas_

2) Matplotlib :   _pip install -U matplotlib_

3) Numpy :  _pip install numpy_

4) Tabulate :   _pip install tabulate_

5) Sklearn :  _pip install -U scikit-learn_

## How to Run the Program

#####  ***The program's code is written completely in Python; A Python Environment will be neccessary to run the program.***

Run command:  _python3 main.py_

Expected Compilation Time:  _~2:30 min_

## Expected Output
PNG files will be created or overwritten inside of folder "Results/" containing scatter plots of each model's data and a bar graph containing final results. Widgets of these graphs will also populate machine, allowing for an interactive view of these graphs.

#### Expected Files:
1) Results/ElasticNet_MVP.png

2) Results/ElasticNet_TOP5.png

3) Results/Lasso_MVP.png

4) Results/Lasso_TOP5.png

5) Results/LinearRegression_MVP.png

6) Results/LinearRegression_TOP5.png

7) Results/MVPBarGraph.png

8) Results/RandomForestRegressor_MVP.png

9) Results/RandomForestRegressor_TOP5.png

10) Results/Ridge_MVP.png

11) Results/Ridge_TOP5.png

12) Results/SGDRegressor_MVP.png

13) Results/SGDRegressor_TOP5.png




