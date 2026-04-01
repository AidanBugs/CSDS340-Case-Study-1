import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

bestC = np.float64(2.0343428262332774)
bestGamma = np.float64(0.6494608808799401)

train_path = "Data/train.csv"   # Full training data (features + labels)
test_path = "Data/test.csv"        # Test data (features + labels)

train_data = pd.read_csv(train_path).to_numpy()
X_train = train_data[:, 0:-1]
y_train = train_data[:, -1]

final_model = Pipeline([
    ('scaler', StandardScaler()),
    ('svm', SVC(C=bestC, gamma=bestGamma, kernel='rbf', random_state=1))
])

final_model.fit(X_train, y_train)

test_data = pd.read_csv(test_path).to_numpy()
X_test = test_data[:, 0:-1]
y_test = test_data[:, -1]

y_pred = final_model.predict(X_test)
test_acc = accuracy_score(y_test, y_pred)

print(f"Test Accuracy: {test_acc * 100:.2f}%")
