
from sklearn.datasets import load_iris

# 1. Load Iris dataset
iris = load_iris()
X = iris.data
y = iris.target

# 2. Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 3. Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# 4. Test different solvers
solvers = ['lbfgs', 'liblinear', 'newton-cg', 'newton-cholesky', 'sag', 'saga']
results = []

for solver in solvers:
    try:
        model = LogisticRegression(solver=solver, max_iter=1000, multi_class='auto')
        model.fit(X_train, y_train)
        train_acc = accuracy_score(y_train, model.predict(X_train))
        test_acc = accuracy_score(y_test, model.predict(X_test))
        results.append([solver, train_acc, test_acc])
    except Exception as e:
        print(f"Solver {solver} failed: {e}")

# Compare results
results_df = pd.DataFrame(results, columns=['Solver', 'Train_Accuracy', 'Test_Accuracy'])
print(results_df)
