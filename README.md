# Used Car Price Prediction Model Using Random Forest Regression

## Introduction

This project explores the development of various regression models to predict the price of used cars based on parameters such as manufacturer, model, condition, cylinders, odometer, size, type, paint color, fuel type, transmission type, title status, and drive type. The final model utilizes a Random Forest Regression algorithm. 

### What is Random Forest Regression?

Random Forest Regression is a machine learning technique that uses multiple decision trees to make predictions. It works by building many trees on random subsets of the data and averaging their predictions to improve accuracy and reduce overfitting. This method helps capture complex patterns in data, making it useful for tasks like predicting continuous values such as prices.

## Model Descriptions

### Linear Regression Model:

Linear regression models are a simple machine learning technique used to predict a continuous target variable by finding the best-fitting straight line through the data. It assumes a linear relationship between the input features and the target variable. In this case it was not a good fit with the data due to a high variability between values as indicated by the MAE as well as a low R-squared value.

```
    Mean Absolute Error: 7136.304877580258
    R-squared: 0.28469741411462746
```

### Tuned Random Forest Regression Model:

In a Random Forest Regressor (RFR), tuning hyperparameters like those in param_grid helps improve model performance. 

```
param_grid = {
    'n_estimators': [100, 200, 500, 1000],         
    'max_depth': [None, 10, 20, 30, 40],           
    'min_samples_split': [2, 5, 10],               
    'min_samples_leaf': [1, 2, 4],                 
    'bootstrap': [True, False]                     
}

random_search = RandomizedSearchCV(estimator=rf_model, param_distributions=param_grid, n_iter=10, cv=3, verbose=2, random_state=42, n_jobs=-1)
```

For example, n_estimators controls the number of trees in the forest; increasing this can improve accuracy but may slow down training. max_depth limits the depth of each tree, preventing overfitting by controlling complexity. min_samples_split and min_samples_leaf ensure each tree branch has enough data points before splitting, helping avoid overfitting. bootstrap decides whether to use bootstrapped datasets for each tree, affecting model variance. The RandomizedSearchCV method randomly tests different combinations of these hyperparameters to find the best configuration for the model.

In this case however, the difference between the tuned and untuned RF models was marginal; in fact the untuned version performed slightly more accurately.

```
    Mean Absolute Error: 2772.539582685514
    R-squared: 0.7894164441139357
```

### Gradient Boosting Regression Model:

Gradient boosting is a more advanced method that builds an ensemble of weak learners (typically decision trees) sequentially, where each new model corrects the errors made by the previous ones, leading to a powerful predictive model. The results were better than the Linear Regression model as well as the tuned RF model; however, there was still a high MAE and comparatively low R-squared.

```
    Mean Absolute Error: 4520.31775355424
    R-squared: 0.6154145922716587
```

### Random Forest Regression Model:

This model made the best price predictions with an R-squared value of nearly 0.8 and the lowest MAE, and was deployed as the final model.

```
    Mean Absolute Error: 2681.001227713523
    R-squared: 0.7931235631444478
```

## Usage + Results


