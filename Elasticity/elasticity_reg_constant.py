import pandas as pd
import numpy as np
import statsmodels.api as sm
from loguru import logger


def compute_constant_elasticity(df: pd.DataFrame, feature1, feature2, target):
    logger.info("Estimating constant elasticity...")
    # Extract means of all features
    mean_feature1 = df[feature1].mean()
    mean_feature2 = df[feature2].mean()

    # Define independent variables (X) and dependent variable (Y)
    X = df[[feature1, feature2]]  # Independent variables
    X = sm.add_constant(X)  # Add constant term for intercept
    Y = df[target]  # Dependent variable

    # Fit the OLS model
    model = sm.OLS(Y, X).fit()

    # Extract the coefficients
    b_feature1 = model.params[feature1]
    b_feature2 = model.params[feature2]
    const = model.params["const"]

    # Predicted target value at feature means
    predicted_at_means = mean_feature1 * b_feature1 + mean_feature2 * b_feature2 + const

    eyex_feature1_mean = (mean_feature1 * b_feature1) / predicted_at_means
    eyex_feature2_mean = (mean_feature2 * b_feature2) / predicted_at_means

    # Display the results as a string
    result = (
        f"{feature1}: elasticity at means = {eyex_feature1_mean:.8f}\n"
        f"{feature2}: elasticity at means = {eyex_feature2_mean:.8f}"
    )

    return result, eyex_feature1_mean, eyex_feature2_mean


if __name__ == "__main__":
    result_text, elasticity1, elasticity2 = compute_constant_elasticity(
        pd.read_stata("./auto_stata.dta"), "weight", "length", "mpg"
    )
    print(result_text)
