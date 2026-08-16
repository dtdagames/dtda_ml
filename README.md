# dtda_ml
[![tests](https://github.com/dtdagames/dtda_ml/actions/workflows/tests.yml/badge.svg)](https://github.com/dtdagames/dtda_ml/actions/workflows/tests.yml)

DTDA ML allows you to run machine learning models like KNN, Linear Regression, Logistic Regression, SVM


5 models are currently available:
- KNN
- Linear Regression
- Logistic Regression
- SVM
- Decision Tree

All of them can be scored with the usual metrics, and saved to a JSON file to be reloaded later.


=== Running the tests ===

The repository is also a small Godot project, so you can open it directly and press F6 on addons/dtda_ml/examples/examples_scene.tscn to see every model run.

The test suite lives in tests/ and needs no framework. Run it headless from the project root:
- godot --headless --script res://tests/run_tests.gd

It prints one line per failure and ends with a count, exiting with 0 when everything passes. Some tests exercise the guards of the addon on purpose, so the output contains expected "MLTools: ..." errors: only the FAIL lines and the final count matter.
The same command runs on every push and pull request, against several Godot versions.

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

=== Metrics ===

_get_perf() only knows how to count correct answers. MLTools also carries the usual metrics, which all check the size of both arrays and report an error instead of returning a wrong score.

For classification, the positive label is 1 by default and can be passed as third argument:
- _accuracy(y_pred, y_test) : percentage of correct answers
- _confusion_matrix(y_pred, y_test, positive) : a dictionary with the tp, fp, tn and fn counts
- _precision(y_pred, y_test, positive) : share of the predicted positives that are right, from 0 to 1
- _recall(y_pred, y_test, positive) : share of the real positives that were found, from 0 to 1
- _f1_score(y_pred, y_test, positive) : harmonic mean of precision and recall

For regression, use these rather than _get_perf(), which binarizes at 0.5 and only makes sense on a binary target:
- _mse(y_pred, y_test) and _rmse(y_pred, y_test) : squared error, the RMSE being in the unit of the target
- _mae(y_pred, y_test) : mean absolute error, less sensitive to outliers
- _r2_score(y_pred, y_test) : share of the variance explained, 1.0 is a perfect fit and a model worse than always answering the mean scores below 0

Example:
- var y_pred = logreg._predict(X_test)
- print("F1: ", ml._f1_score(y_pred, y_test))
- print("Confusion: ", ml._confusion_matrix(y_pred, y_test))
- print("R2: ", ml._r2_score(linreg._predict(X_test), y_test))

=== Scaler ===

The models scale their features on their own, so you don't need DTDAScaler to use them. It is there for your own data, and it is what the models use internally.

Use DTDAScaler.new() for a standardization (each column centered on its mean and divided by its standard deviation), or DTDAScaler.new(DTDAScaler.MINMAX) to bring each column into [0, 1]. A constant column is left alone instead of dividing by zero.
_fit() learns the scaling, _transform() applies it, _fit_transform() does both, and _inverse_transform() brings values back to the original unit.

Fit the scaler on your training set only, then apply that same scaler to the test set: refitting on the test set would scale it differently and quietly ruin your predictions.

Example:
- var scaler = DTDAScaler.new()
- var X_train_scaled = scaler._fit_transform(X_train)
- var X_test_scaled = scaler._transform(X_test) #the scaling learned on X_train
- print("Back to the original unit: ", scaler._inverse_transform(X_train_scaled))

=== Saving and loading a model ===

Every model can be written to a JSON file and read back, so you can train once and ship the weights with your game instead of retraining at every launch.
_save(path) returns true on success, _load(path) fills a model you just created. Both report a clear error and return false on failure, and _load() refuses a file holding a different kind of model.

Use a user:// path: res:// is read only once the game is exported.

Example:
- var linreg = DTDALinReg.new(0.01, 1000)
- linreg._fit(X_train, y_train)
- linreg._save("user://linreg.json")
- var loaded = DTDALinReg.new(0.01, 1000) #a brand new model, never fitted
- if loaded._load("user://linreg.json"):
-     print("Prediction: ", loaded._predict(X_test)) #same results as the saved model

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
Like Linear Regression, the features are standardized internally by _fit(), so you don't have to scale them yourself.

Example:
- var logreg = DTDALogReg.new(0.01, 1000)
- logreg._fit(X_train, y_train)
- var X_test = [
    [1, 1, 0, 1]
  ]
- print("Logistic Regression prediction: ", logreg._predict(X_test))

=== SVM Model ===

Use DTDASVM.new() to create a new model. _fit() and _predict() allows you to train and use the model. This model is only for classification (1 or -1).
The features are standardized internally by _fit() as well. A point sitting exactly on the decision boundary is predicted as 1.

Example:
- var svm = DTDASVM.new(0.01, 0.01, 1000)
- svm._fit(X_train, y_train)
- var X_test = [
    [1, 1, 0, 1]
  ]
- print("SVM prediction: ", svm._predict(X_test))

=== Decision Tree Model ===

Use DTDATree.new() to create a new model. _fit() and _predict() allows you to train and use the model. It is the only model here handling a non linear frontier: it separates a XOR, which the linear models cannot.

DTDATree.new(max_depth, min_samples_split, mode) takes:
- max_depth : how deep the tree may grow, 5 by default. The main guard against overfitting
- min_samples_split : a node holding fewer rows than this becomes a leaf, 2 by default
- mode : DTDATree.CLASSIFIER (the default) splits on the Gini impurity and a leaf answers the majority label, DTDATree.REGRESSOR splits on the variance and a leaf answers the mean

A tree compares each feature to a threshold, so the scale of your data does not matter: unlike the other models it does no scaling at all, and none is needed.
Being made of thresholds, it also answers a constant outside the range it was trained on, where a linear regression keeps extrapolating.

Example:
- var tree = DTDATree.new(3, 2, DTDATree.CLASSIFIER)
- tree._fit(X_train, y_train)
- var X_test = [
    [1, 1, 0, 1]
  ]
- print("Tree prediction: ", tree._predict(X_test))
- var regressor = DTDATree.new(3, 2, DTDATree.REGRESSOR) #same model, on a continuous target
- regressor._fit(X_train, y_train)
