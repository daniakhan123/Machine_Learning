
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# 1. Load dataset
data = pd.read_csv("heart_disease.csv")  # replace with your CSV
print(data.head())
print(data.info())
print(data.describe())

# 2. Data Wrangling
# Check missing values
print(data.isnull().sum())

# Example: Fill missing with median
data = data.fillna(data.median())

# 3. Select features and target
X = data.drop('target', axis=1)  # replace 'target' with your label column
y = data['target']

# 4. Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 5. Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# 6. Logistic Regression with different penalties
penalties = ['l1', 'l2', 'elasticnet']
solver_map = {'l1':'liblinear', 'l2':'lbfgs', 'elasticnet':'saga'}
results = []

for penalty in penalties:
    if penalty == 'elasticnet':
        model = LogisticRegression(penalty=penalty, solver=solver_map[penalty], l1_ratio=0.5, max_iter=1000)
    else:
        model = LogisticRegression(penalty=penalty, solver=solver_map[penalty], max_iter=1000)
    
    model.fit(X_train, y_train)
    train_acc = accuracy_score(y_train, model.predict(X_train))
    test_acc = accuracy_score(y_test, model.predict(X_test))
    results.append([penalty, train_acc, test_acc])
    print(f"\nPenalty: {penalty}")
    print("Training Accuracy:", train_acc)
    print("Testing Accuracy:", test_acc)
    print(classification_report(y_test, model.predict(X_test)))

# Compare results in DataFrame
results_df = pd.DataFrame(results, columns=['Penalty','Train_Accuracy','Test_Accuracy'])
print(results_df)
