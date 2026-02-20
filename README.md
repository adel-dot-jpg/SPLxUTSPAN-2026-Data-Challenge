## Methodology and Overview
This notebook utilizes an XGBoost regression model. The objective is to predict the scaled angle, depth, and left right deviation of the shot. The full code is available on my public GitHub repository [**here**](https://github.com/adel-dot-jpg/SPLxUTSPAN-2026-Data-Challenge)

## Feature Extraction
The raw dataset provides spatial coordinates stored as string representations of lists. To process this, my pipeline uses the ast library (abstract syntax tree) to safely evaluate these strings into NumPy arrays. Because a basketball shot varies in duration and frame count, extracting summary statistics is simpler than feeding raw time series data. For every joint coordinate across the entire duration of the shot, we calculate the mean and standard deviation. This flattens the variable length sequences into a fixed length tabular format suitable for tree based models, capturing both the average position and the physical variability of the shooter's movement.

## Model Strategy and Validation
For the predictive model, I utilize XGBoost regressor wrapped in a MultiOutputRegressor to handle the three continuous target variables simultaneously. Given the small sample size of 458 shots across only 5 participants, if the model sees the same participant in both the training and validation sets, there is the risk of the model memorizing player body mechanics instead of shot physics. To prevent this, Group K Fold cross validation, grouping by participant ID is employed to validate output consistency. This checks if the model is learning generalized shooting mechanics and evaluates its performance on an entirely unseen participant during each fold.

## Final Prediction and Scaling
After validating the model architecture, it is retrained on the entire training dataset to maximize the amount of data it learns from before inferring on the test set. Finally, the raw predictions must be scaled to adhere to the competition requirements. A MinMax scaling formula is applied using the provided bounds for angle, depth, and left right deviation, clipping the final results between 0 and 1 to generate the final submission file.
