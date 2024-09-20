# dtda_ml
DTDA ML allows you to run machine learning models like linear regression, logistic regression, KNN and SVM


3 models are currently available:
- KNN
- Linear Regression
- Logistic Regression

=== KNN Model ===
Use DTDAKNN.new() to create a new model. _fit and _predict allows you to train and use the model
This model is better for classficiation
Example:
  var knn = DTDAKNN.new(3)
  knn._fit(X_train, Y_train)
  print("Knn prediction: ", knn._predict(X_test))

=== Linear Regression Model ===
Use DTDALinReg.new() to create a new model. _fit and _predict allows you to train and use the model
This model is better for Regression
Example:
  var linreg = DTDALinReg.new(0.01, 1000)
	linreg._fit(X_train, Y_train)
	print("Linear Regression prediction: ", linreg._predict(X_test))

=== Linear Regression Model ===
Use DTDALogReg.new() to create a new model. _fit and _predict allows you to train and use the model
This model is only for classficiation (-1 or 1)
Example:
  var logreg = DTDALogReg.new(0.01, 1000)
	logreg._fit(X_train, Y_train)
	print("Logistic Regression prediction: ", logreg._predict(X_test))
