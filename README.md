# dtda_ml
DTDA ML allows you to run machine learning models like KNN, Linear Regression, Logistic Regression, SVM


3 models are currently available:
- KNN
- Linear Regression
- Logistic Regression


=== MLTools features ===

Use MLTools.new() to create a new MLTools. _dropVariable() and _getVariable() allows you to drop a column, or keep column from an array.
This is usefull to create X_train and Y_train for all models

Example:
- data = [
    [1, 1, 1, 0, 1],
    [1, 1, 1, 1, 1],
    [1, 0, 0, 0, 0]
  ]
- var ml = MLTools.new()
- var X_train = ml._dropVariable(data, data[0].size()-1) #return an array of array without the last column
- var Y_train = ml._getVariable(data, data[0].size()-1) #return an array of array only with the last column

=== KNN Model ===

Use DTDAKNN.new() to create a new model. _fit() and _predict() allows you to train and use the model. This model is better for classification.

Example:
- var knn = DTDAKNN.new(3)
- knn._fit(X_train, Y_train)
- var X_test = [
    [1, 1, 0, 1]
  ]
- print("Knn prediction: ", knn._predict(X_test))

=== Linear Regression Model ===

Use DTDALinReg.new() to create a new model. _fit() and _predict() allows you to train and use the model. This model is better for Regression.

Example:
- var linreg = DTDALinReg.new(0.01, 1000)
- linreg._fit(X_train, Y_train)
- var X_test = [
    [1, 1, 0, 1]
  ]
- print("Linear Regression prediction: ", linreg._predict(X_test))

=== Logistic Regression Model ===

Use DTDALogReg.new() to create a new model. _fit() and _predict() allows you to train and use the model. This model is only for classification (1 or 0).

Example:
- var logreg = DTDALogReg.new(0.01, 1000)
- logreg._fit(X_train, Y_train)
- var X_test = [
    [1, 1, 0, 1]
  ]
- print("Logistic Regression prediction: ", logreg._predict(X_test))


