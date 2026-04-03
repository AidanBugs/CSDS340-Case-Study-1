---
title: "Group 8 Case Study 1: Introduction to Machine Learning CSDS340"
author: 
  - name: "Aidan Bugayong"
    affiliations:
      name: "Department of Computer and Data Sciences, Case Western Reserve University"
      city: "Cleveland"
      state: "Ohio"
  - name: "Sam Lin"
    affiliations:
      name: "Department of Computer and Data Sciences, Case Western Reserve University"
      city: "Cleveland"
      state: "Ohio"
date: "2026-04-03"
title-block-banner: true
execute:
  echo: true
  warning: false
format: 
  pdf: 
    output-file: "group8_CS1.pdf"
    documentclass: article
    geometry:
      - top=30mm
      - left=20mm
      - right=20mm
      - heightrounded
    mainfont: Times New Roman
    fontsize: 12pt
    colorlinks: true
    date-format: long
---

\pagebreak


# Functions and Parameters

We used RandomizedSearchCV to explore hyperparameters for each classifier. Specifically we varied the specific functions and parameters as shown below:

Logistic Regression (LogisticRegression)

- Penalty (Regularization Type): 'l1' or 'l2'
- Regularization strength (C): sampled from a uniform distribution Uniform(0.01, 10)
- Solver (solver): 'liblinear' or 'saga’

SVM (Support Vector Machine) with RBF kernel (SVC)

- Regularization (C): Uniform(0.1, 10), trade-off between margin maximization and misclassification penalty.
- Kernel coefficient (gamma): Uniform(0.01, 1), defines the influence of a single training example.
- Kernel (kernel): fixed to 'rbf' (non-linear mapping) since it performs in general better than the other kernels. 

Decision Tree (DecisionTreeClassifier)

- Maximum depth (max_depth): randint(3, 20), prevents overfitting by limiting tree growth.
- Minimum samples to split a node (min_samples_split): randint(2, 20), minimum number of samples required to split an internal node.
- Minimum samples at a leaf (min_samples_leaf): randint(1, 10), ensures leaves have enough samples.
- Split criterion (impurity measures): 'gini' or 'entropy'

All classifiers share the common preprocessing step StandardScaler (centering and scaling features) in every pipeline.

# Explain Data Preprocessing:

Validation Set:
- Train/Test Split: 15% holdout with stratification (stratify=y) to preserve class distribution – essential for imbalanced data. This was used as a final validation test for choosing the best model for generalizations.

Standard Scalar:
- We conducted the standard scalar in each pipeline as it serves a crucial role to ensure models do not get biased toward features with larger ranges. Models like SVM and logistic regression could be especially affected by unscaled dataset. 

PCA:
- PCA is the process of projecting features onto the direction of maximum variance. The tuned parameter is sampled as a fraction between 0.5 to 1, which stands for the amount of variance the parameter explained. This procedure effectively reduces dimensionality, reduces collinearity, and also reduces overfitting. In our model, applying the PCA has led to higher accuracy in all cases.

LDA:
- Different from PCA, LDA is a supervised approach where it maximizes the between class separation relative to the within class scatter. In binary classification, since there are only two results, the maximum number of discriminating components is 1. The shrinkage parameter regularizes the within class parameter within class covariance. This method did not work as well, resulting in accuracy lower than both PCA and no dimension reduction. 

Standard Scaler, PCA, and LDA were all used to train separate models and the best performing model (separated by preprocessing type) was then used for the final testing.

# Explain Hyper Parameter Tuning Process 
The hyper parameter tuning process uses randomized search to try finding the optimal hyper parameter tuple. This process was chosen over an exhaustive grid search because we figured that our peers would default to using grid search, resulting in many groups having identical optimal models. In order to attempt to be in the higher performing quartiles, the risk of using a randomized search must be taken in order to potentially have a more unique higher performing model.

On top of the randomized search, many of the hyper parameters were chosen at uniform random (integer for decision tree parameters). Our group did not use any distributions of the parameter sampling except for the uniform distribution.

Scoring with respect to accuracy was performed on a stratified k-fold of k=10, resulting in similar class proportions on each fold and 200 random parameter tuples per model. Since we tested for different preprocessing techniques (standard scaler, PCA, LDA) each of those were also part of the different model training process. Thus for each model, we trained 200*10*3=6000 models and found that the highest performing model was the SVM with the following hyper parameters:


SVM:

- C=2.0343428262332774
- 𝛾=0.6494608808799401

## Why our Tuning Process Works

As mentioned before, random search is a good heuristic for finding optimal hyper parameters because it doesn’t stick to a strict grid or require an exhaustive search. The stratified k-fold cross validation ensures similar distribution of class 0 and class 1 which increases the robustness of the fold models. Scaling is present in all 3 preprocessing types, which is important for the different parameters and models to ensure further robustness. Additionally since scoring was with respect to accuracy in the final test done by TA’s, this was the method used to refit models after each search. 

# The reason about your final choice of classification model and its related data preprocessing and hyperparameter tuning algorithms
We choose the SVM with parameters, C=2.0343428262332774, 𝛾=0.6494608808799401 and preprocessing of just a standard scalar. We chose this model because it had the highest accuracy in our separate validation set which was withheld during the k-fold process. These parameters ended up having the highest accuracy for the separate validation set out of all of the models.

# Insightfulness and clarity of your observations and discussions. (Please be free to add the approaches you tried but failed before arriving at the best solution.)
Other attempts on models include trying different preprocessing techniques with dimensionality reduction, specifically using LDA and PCA. Of the two PCA performed very similarly to no dimensionality reduction but we found that LDA had a significant decrease in accuracy. We also tried different k values for k-folds and different random states to end up with different hyper parameters which were tested further. Additionally, we varied the random search iterations from an initial 20 to our final 200 amount resulting in a large amount of hyper parameter tuples tested on.

We also attempted a weighted ensemble method with the top 10 performing models (from the 200 splits) and the weights applied to each model were their accuracy during validation. Thus, when we tested on our separate test set (15% of our training) all 10 models would vote and we’d use a weighted sum of the models to determine the output. This unfortunately slightly decreased the performance compared to only using the top performing model in each category (this was used for the three model types and the three different preprocessing techniques so 9 different ensemble models). As a result we proceeded forward with a single top performing model of SVM with the parameters stated above. 
