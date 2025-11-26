from sklearn.linear_model import Perceptron

# 1. Train Perceptron
perceptron = Perceptron(max_iter=1000, random_state=42)
perceptron.fit(X_train, y_train)

# 2. Evaluate
train_acc = accuracy_score(y_train, perceptron.predict(X_train))
test_acc = accuracy_score(y_test, perceptron.predict(X_test))

print("Perceptron Training Accuracy:", train_acc)
print("Perceptron Testing Accuracy:", test_acc)

# Compare with Logistic Regression
lr_model = LogisticRegression(solver='lbfgs', max_iter=1000, multi_class='auto')
lr_model.fit(X_train, y_train)
print("Logistic Regression Training Accuracy:", accuracy_score(y_train, lr_model.predict(X_train)))
print("Logistic Regression Testing Accuracy:", accuracy_score(y_test, lr_model.predict(X_test)))
