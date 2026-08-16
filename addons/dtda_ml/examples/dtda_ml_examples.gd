extends Node

var mltools

var dataKNN = [
	[2, 4, 2, 1, 0, 0, 3],
	[2, 2, 4, 0, 0, 0, 4],
	[4, 2, 1, 1, 0, 1, 5],
	[2, 2, 4, 0, 1, 1, 6],
]

var dataLinR = [
	[1.6, 40000],
	[4.6, 60000],
	[4.2, 58000],
	[4.1, 59000],
	[5.4, 80000],
	[8.1, 100000],
	[8.9, 110000],
	[9.2, 110000],
	[9.3, 114000],
	[10.2, 121000],
]

var dataLogR = [
	[2, 4, 2, 1, 0, 0, 0],
	[2, 2, 4, 0, 0, 0, 0],
	[4, 2, 1, 1, 0, 1, 1],
	[2, 2, 4, 0, 1, 1, 1],
]

var dataSVM = [
	[2, 4, 2, 1, 0, 0, 0],
	[2, 2, 4, 0, 0, 0, 0],
	[4, 2, 1, 1, 0, 1, 1],
	[2, 2, 4, 0, 1, 1, 1],
]

func _ready():
	mltools = MLTools.new()
	
	_knn_example()
	_linreg_example()
	_logreg_example()
	_svm_example()
	_tree_example()
	_scaler_example()
	_metrics_example()
	_persistence_example()

func _knn_example():
	var X_train = mltools._dropVariable(dataKNN, dataKNN[0].size()-1)
	var y_train = mltools._getVariable(dataKNN, dataKNN[0].size()-1)
	var X_test = [
		[1, 4, 1, 1, 0, 0],
		[2, 2, 4, 1, 1, 1],
		[4, 1, 1, 0, 1, 0],
	]
	var y_test = [
		3,
		6,
		5,
	]
	
	var knn = DTDAKNN.new(3)
	knn._fit(X_train, y_train)
	print("KNN predictions: ", knn._predict(X_test))
	print("KNN score: ", mltools._get_perf(knn._predict(X_test), y_test, 0), "%")

func _linreg_example():
	var X_train = mltools._dropVariable(dataLinR, dataLinR[0].size()-1)
	var y_train = mltools._getVariable(dataLinR, dataLinR[0].size()-1)
	var X_test = [
		[7.2],
		[9.0],
		[11.1],
	]
	
	var linreg = DTDALinReg.new(0.01, 1000)
	linreg._fit(X_train, y_train)
	print("Linear Regression predictions: ", linreg._predict(X_test))

func _logreg_example():
	var X_train = mltools._dropVariable(dataLogR, dataLogR[0].size()-1)
	var Y_train = mltools._getVariable(dataLogR, dataLogR[0].size()-1)
	var X_test = [
		[1, 3, 1, 0, 1, 0],
		[2, 2, 4, 1, 1, 1],
		[4, 1, 1, 0, 1, 0],
	]
	var y_test = [
		0,
		1,
		1,
	]
	
	var logreg = DTDALogReg.new(0.01, 1000)
	logreg._fit(X_train, Y_train)
	print("Logistic Regression predictions: ", logreg._predict(X_test))
	print("Logistic Regression score: ", mltools._get_perf(logreg._predict(X_test), y_test, 2), "%")

func _svm_example():
	var X_train = mltools._dropVariable(dataSVM, dataSVM[0].size()-1)
	var Y_train = mltools._getVariable(dataSVM, dataSVM[0].size()-1)
	var X_test = [
		[1, 3, 1, 0, 1, 0],
		[2, 2, 4, 1, 1, 1],
		[4, 1, 1, 0, 1, 0],
	]
	var y_test = [
		0,
		1,
		1,
	]
	
	var svm = DTDASVM.new(0.01, 0.01, 1000)
	svm._fit(X_train, Y_train)
	print("SVM predictions: ", svm._predict(X_test))
	print("SVM score: ", mltools._get_perf(svm._predict(X_test), y_test, 3), "%")

func _tree_example():
	# classification, the tree answers the majority label of the leaf
	var X_train = mltools._dropVariable(dataLogR, dataLogR[0].size()-1)
	var Y_train = mltools._getVariable(dataLogR, dataLogR[0].size()-1)
	var X_test = [
		[1, 3, 1, 0, 1, 0],
		[2, 2, 4, 1, 1, 1],
		[4, 1, 1, 0, 1, 0],
	]
	var y_test = [0, 1, 1]

	var tree = DTDATree.new(3, 2, DTDATree.CLASSIFIER)
	tree._fit(X_train, Y_train)
	print("Tree predictions: ", tree._predict(X_test))
	print("Tree score: ", mltools._accuracy(tree._predict(X_test), y_test), "%")

	# regression, the tree answers the mean of the leaf
	var X_lin = mltools._dropVariable(dataLinR, dataLinR[0].size()-1)
	var y_lin = mltools._getVariable(dataLinR, dataLinR[0].size()-1)
	var regressor = DTDATree.new(3, 2, DTDATree.REGRESSOR)
	regressor._fit(X_lin, y_lin)
	print("Tree regression: ", regressor._predict([[7.2], [9.0], [11.1]]))
	print("Tree R2: ", mltools._r2_score(regressor._predict(X_lin), y_lin))

	# a XOR, which no linear model can separate
	var xor_X = [[0, 0], [0, 1], [1, 0], [1, 1]]
	var xor_y = [0, 1, 1, 0]
	var xor_tree = DTDATree.new(4, 2, DTDATree.CLASSIFIER)
	xor_tree._fit(xor_X, xor_y)
	print("Tree on a XOR: ", xor_tree._predict(xor_X), " expected ", xor_y)

# the models scale their features on their own, use DTDAScaler for your own data
func _scaler_example():
	var raw = [
		[1.6, 40000],
		[5.4, 80000],
		[10.2, 121000],
	]

	var standard = DTDAScaler.new()
	print("Standardized: ", standard._fit_transform(raw))

	var minmax = DTDAScaler.new(DTDAScaler.MINMAX)
	var scaled = minmax._fit_transform(raw)
	print("Min-max: ", scaled)
	print("Back to the original unit: ", minmax._inverse_transform(scaled))

func _metrics_example():
	# classification, on the logistic regression data
	var X_train = mltools._dropVariable(dataLogR, dataLogR[0].size()-1)
	var Y_train = mltools._getVariable(dataLogR, dataLogR[0].size()-1)
	var X_test = [
		[1, 3, 1, 0, 1, 0],
		[2, 2, 4, 1, 1, 1],
		[4, 1, 1, 0, 1, 0],
	]
	var y_test = [0, 1, 1]

	var logreg = DTDALogReg.new(0.01, 1000)
	logreg._fit(X_train, Y_train)
	var y_pred = logreg._predict(X_test)
	print("Accuracy: ", mltools._accuracy(y_pred, y_test), "%")
	print("Confusion matrix: ", mltools._confusion_matrix(y_pred, y_test))
	print("Precision: ", mltools._precision(y_pred, y_test))
	print("Recall: ", mltools._recall(y_pred, y_test))
	print("F1 score: ", mltools._f1_score(y_pred, y_test))

	# regression, scored on the training set itself
	var X_lin = mltools._dropVariable(dataLinR, dataLinR[0].size()-1)
	var y_lin = mltools._getVariable(dataLinR, dataLinR[0].size()-1)
	var linreg = DTDALinReg.new(0.01, 1000)
	linreg._fit(X_lin, y_lin)
	var lin_pred = linreg._predict(X_lin)
	print("R2: ", mltools._r2_score(lin_pred, y_lin))
	print("RMSE: ", mltools._rmse(lin_pred, y_lin))
	print("MAE: ", mltools._mae(lin_pred, y_lin))

# train once, ship the weights, predict without the training set
func _persistence_example():
	var path = "user://dtda_linreg.json"
	var X_train = mltools._dropVariable(dataLinR, dataLinR[0].size()-1)
	var y_train = mltools._getVariable(dataLinR, dataLinR[0].size()-1)
	var X_test = [
		[7.2],
		[9.0],
		[11.1],
	]

	var linreg = DTDALinReg.new(0.01, 1000)
	linreg._fit(X_train, y_train)
	print("Before saving: ", linreg._predict(X_test))
	if not linreg._save(path):
		return

	# a brand new model, never fitted
	var loaded = DTDALinReg.new(0.01, 1000)
	if loaded._load(path):
		print("After loading: ", loaded._predict(X_test))
