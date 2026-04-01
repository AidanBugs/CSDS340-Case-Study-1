import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.model_selection import (RandomizedSearchCV, StratifiedKFold,
                                     RepeatedStratifiedKFold, train_test_split)
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from scipy.stats import uniform, randint
from sklearn.metrics import accuracy_score, f1_score

# -------------------------
# 1. Load and prepare data
# -------------------------
path = "./Data/train.csv"
data = pd.read_csv(path).to_numpy()
X = data[:, 0:-1]   # features
y = data[:, -1]     # label

path_test = "./Data/test.csv"
if os.path.exists(path_test):
    print("Test file found. Using provided test set.")
    test_data = pd.read_csv(path_test).to_numpy()
    X_test = test_data[:, 0:-1]
    y_test = test_data[:, -1]
    X_train = X
    y_train = y
else:
    print("Test file not found. Splitting training data (85% train, 15% test).")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.15, stratify=y, random_state=1
    )


# -------------------------
# 2. Define cross-validation strategy (for hyperparameter tuning)
# -------------------------
cv_strategy = StratifiedKFold(n_splits=10, shuffle=True, random_state=1)

# -------------------------
# 3. Hyperparameter distributions for each classifier
#    Keys are descriptive names; we map to actual classes later.
# -------------------------
param_distributions = {
    'LogisticRegression': {
        'clf__C': uniform(loc=0.01, scale=10),
        'clf__penalty': ['l1', 'l2'],
        'clf__solver': ['liblinear', 'saga']
    },
    'SVM': {
        'clf__C': uniform(loc=0.1, scale=10),
        'clf__gamma': uniform(loc=0.01, scale=1),
        'clf__kernel': ['rbf']
    },
    'DecisionTree': {
        'clf__max_depth': randint(3, 20),
        'clf__min_samples_split': randint(2, 20),
        'clf__min_samples_leaf': randint(1, 10),
        'clf__criterion': ['gini', 'entropy']
    }
}

# Mapping from descriptive names to actual classifier classes
classifier_map = {
    'LogisticRegression': LogisticRegression,
    'SVM': SVC,
    'DecisionTree': DecisionTreeClassifier
}

# Hyperparameter distributions for dimensionality reduction steps
pca_param_dist = {
    'pca__n_components': uniform(0.5, 0.5)   # fraction of variance explained between 0.5 and 1.0
}

lda_param_dist = {
    'lda__n_components': randint(1, 2),           # for binary classification, max components = 1 (or use None)
    'lda__solver': ['svd', 'lsqr', 'eigen'],      # solvers; 'svd' doesn't require shrinkage
    'lda__shrinkage': uniform(0.0, 1.0)           # only used with 'lsqr' or 'eigen' solvers
}

# -------------------------
# 4. Build pipelines: no reduction, with PCA, with LDA
# -------------------------
def build_pipelines():
    """Create three pipelines for each classifier: no reduction, PCA, LDA."""
    pipelines = {}
    for name, param_grid in param_distributions.items():
        clf_class = classifier_map[name]

        # 1. Pipeline without dimensionality reduction
        pipe_no_red = Pipeline([
            ('scaler', preprocessing.StandardScaler()),
            ('clf', clf_class(random_state=1))
        ])
        pipelines[f"{name}_noRed"] = (pipe_no_red, param_grid)

        # 2. Pipeline with PCA
        pipe_pca = Pipeline([
            ('scaler', preprocessing.StandardScaler()),
            ('pca', PCA(random_state=1)),
            ('clf', clf_class(random_state=1))
        ])
        # Merge classifier parameters with PCA parameters
        param_grid_pca = {**param_grid, **pca_param_dist}
        pipelines[f"{name}_PCA"] = (pipe_pca, param_grid_pca)

        # 3. Pipeline with LDA
        pipe_lda = Pipeline([
            ('scaler', preprocessing.StandardScaler()),
            ('lda', LDA()),
            ('clf', clf_class(random_state=1))
        ])
        # Merge classifier parameters with LDA parameters
        param_grid_lda = {**param_grid, **lda_param_dist}
        pipelines[f"{name}_LDA"] = (pipe_lda, param_grid_lda)

    return pipelines

pipelines = build_pipelines()

# -------------------------
# 5. Run RandomizedSearchCV for each pipeline on TRAINING set
# -------------------------
best_models = {}
cv_results = {}

for name, (pipe, param_grid) in pipelines.items():
    print(f"\n=== Tuning {name} ===")
    search = RandomizedSearchCV(
        pipe,
        param_distributions=param_grid,
        n_iter=200, 
        cv=cv_strategy,
        scoring={'accuracy': 'accuracy', 'f1': 'f1'},
        refit='accuracy',
        verbose=1,
        random_state=1,
        n_jobs=-1
    )
    # Fit on training data only
    search.fit(X_train, y_train)

    best_models[name] = search.best_estimator_
    cv_results[name] = {
        'best_params': search.best_params_,
        'best_cv_accuracy': search.best_score_,
        'best_cv_f1': search.cv_results_['mean_test_f1'][search.best_index_]
    }

    print(f"Best parameters: {search.best_params_}")
    print(f"Best CV accuracy: {cv_results[name]['best_cv_accuracy']:.4f}")
    print(f"Best CV F1 score: {cv_results[name]['best_cv_f1']:.4f}")

# -------------------------
# 6. Evaluate each best model on the TEST set
# -------------------------
print("\n=== Test Set Evaluation ===")
print(f"{'Pipeline':<25} {'Test Acc':>10} {'Test F1':>10}")
print("-" * 50)

test_results = {}
for name, model in best_models.items():
    y_pred = model.predict(X_test)
    test_acc = accuracy_score(y_test, y_pred)
    test_f1 = f1_score(y_test, y_pred, average='weighted')  # adjust if binary
    test_results[name] = {'accuracy': test_acc, 'f1': test_f1}
    print(f"{name:<25} {test_acc:>10.4f} {test_f1:>10.4f}")

# -------------------------
# 7. Compare results (CV vs Test)
# -------------------------
print("\n=== Comparison of Best Models (CV vs Test) ===")
print(f"{'Pipeline':<25} {'CV Acc':>10} {'Test Acc':>10} {'CV F1':>10} {'Test F1':>10}")
print("-" * 70)
for name in cv_results.keys():
    cv_acc = cv_results[name]['best_cv_accuracy']
    cv_f1 = cv_results[name]['best_cv_f1']
    test_acc = test_results[name]['accuracy']
    test_f1 = test_results[name]['f1']
    print(f"{name:<25} {cv_acc:>10.4f} {test_acc:>10.4f} {cv_f1:>10.4f} {test_f1:>10.4f}")
