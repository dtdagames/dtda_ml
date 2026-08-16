# dtda_ml
DTDA ML allows you to run machine learning models like KNN, Linear Regression, Logistic Regression, SVM


4 models are currently available:
- KNN
- Linear Regression
- Logistic Regression
- SVM


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
- var y_train = ml._getVariable(data, data[0].size()-1) #return an array of array only with the last column

_get_perf() scores a model: it compares the predictions to the expected labels and returns the percentage of correct answers, rounded to 0.01.
It takes the type of the model as third argument, so it can align both arrays before comparing them:
- 0 : KNN, predictions and labels are compared as they are
- 1 : Linear Regression, predictions above 0.5 are read as 1 and the others as 0, so this only makes sense on a binary target
- 2 : Logistic Regression, predictions and labels are compared as they are
- 3 : SVM, the 0 labels are converted to -1 to match what the model predicts

Example:
- var y_pred = knn._predict(X_test)
- var y_test = [3, 6, 5]
- print("KNN score: ", ml._get_perf(y_pred, y_test, 0), "%")

=== KNN Model ===

Use DTDAKNN.new() to create a new model. _fit() and _predict() allows you to train and use the model. This model is better for classification.
The prediction is the most frequent label among the k nearest neighbors. When two labels are tied, the one carried by the closest neighbor wins.

Example:
- var knn = DTDAKNN.new(3)
- knn._fit(X_train, y_train)
- var X_test = [
    [1, 1, 0, 1]
  ]
- print("KNN prediction: ", knn._predict(X_test))

=== Linear Regression Model ===

Use DTDALinReg.new() to create a new model. _fit() and _predict() allows you to train and use the model. This model is better for Regression.
Features and target are standardized internally by _fit(), so the gradient descent stays stable whatever the scale of your data. You don't have to normalize anything beforehand: _predict() gives its results back in the unit of the training target.

Example:
- var linreg = DTDALinReg.new(0.01, 1000)
- linreg._fit(X_train, y_train)
- var X_test = [
    [1, 1, 0, 1]
  ]
- print("Linear Regression prediction: ", linreg._predict(X_test))

=== Logistic Regression Model ===

Use DTDALogReg.new() to create a new model. _fit() and _predict() allows you to train and use the model. This model is only for classification (1 or 0).

Example:
- var logreg = DTDALogReg.new(0.01, 1000)
- logreg._fit(X_train, y_train)
- var X_test = [
    [1, 1, 0, 1]
  ]
- print("Logistic Regression prediction: ", logreg._predict(X_test))

=== SVM Model ===

Use DTDASVM.new() to create a new model. _fit() and _predict() allows you to train and use the model. This model is only for classification (1 or -1).

Example:
- var svm = DTDASVM.new(0.01, 0.01, 1000)
- svm._fit(X_train, y_train)
- var X_test = [
    [1, 1, 0, 1]
  ]
- print("SVM prediction: ", svm._predict(X_test))
