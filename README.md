# Breast Cancer Prediction and Analysis
## Predict whether the cancer is benign or malignant

**Author**: SHAILESH DHAMA

### Business problem: To predict whether the cancer is benign or malignant.

### Dataset : https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+(Diagnostic)

### Required Libraries :

    1.Numpy
    2.Pandas
    3.Matplotlib
    4.Seaborn
    5.scikit-learn
    6.Plotly

## STEPS :

    - Read data
    - Missing values
    - Drop useless variables
    - Exploratory Data Analysis (EDA)
    - Features distribution(hue = diagnosis)
    - Positive correlated features
    - Uncorrelated features
    - Negative correlated features
    - Principal Component Analysis
    - Confusion matrix
    - Precision-recall curve
    - ROC curve
    - Logistic Confusion matrix
    - Logistic regression with RFE
    - Predictive model : Logistic Regression
    - Predictive Model-2: Ensemble Classifier to maximise precision and detect all malignant tumors
    - Voting classifier : select threshold (recall = 100%)
    - Voting classifier : predicting with recall = 100% (precision = 92%)
    - Models performance plot (accuracy, precision, recall)

## RESULTS :

#### Data distribution
![graph1](./BC_0.PNG)
![graph1](./BC-1.PNG)
![graph2](./BC-2.PNG)
> Data distribution.

#### Features distribution(hue = diagnosis)
![graph3](./BC-3.PNG)
![graph4](./BC-4.PNG)
![graph5](./BC-5.PNG)
![graph6](./BC-6.PNG)
![graph7](./BC-7.PNG)
![graph8](./BC-8.PNG)
![graph9](./BC-9.PNG)
> Mean of Features distribution.

#### Correlation Matrix for Variables
![graph10](./BC-10.PNG)
> correlation matrix


#### Positive correlated features
![graph10](./BC_1.png)
> Positive correlated features


#### Uncorrelated features
![graph10](./BC_2.png)
> Uncorrelated features

#### Negative correlated features
![graph10](./BC_3.png)
> Negative correlated features

#### Principal Component Analysis
![graph10](./BC-15.PNG)
> PCA : Explained variance
![graph10](./BC-16.PNG)
> PCA Scatter(2 Components)
![graph10](./BC-17.PNG)
> PCA Scatter(3 Components)

### Final RESULT : Overall performance of Models.
![graph10](./BC_4.png)
> Performance(cross_val_mean)


### For further information:

Please review the narrative of our analysis in [our jupyter notebook](./breast-cancer-analysis-and-prediction-on-winconsin.ipynb)

For any additional questions, please contact **shaileshshettyd@gmail.com)

##### Repository Structure:

```
├── README.md                                                             <- The top-level README for reviewers of this project.
├── breast-cancer-analysis-and-prediction-on-winconsin.ipynb              <- narrative documentation of analysis in jupyter notebook.
└── images                                                                <- both sourced externally and generated from code.
```
