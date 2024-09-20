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

func _ready():
	mltools = MLTools.new()
	
	_knn_example()
	_linreg_example()
	_logreg_example()

func _knn_example():
	var X_train = mltools._dropVariable(dataKNN, dataKNN[0].size()-1)
	var Y_train = mltools._getVariable(dataKNN, dataKNN[0].size()-1)
	var X_test = [
		[1, 3, 1, 0, 1, 0],
		[2, 2, 4, 1, 1, 1],
		[4, 1, 1, 0, 1, 0],
	]
	
	var knn = DTDAKNN.new(6)
	knn._fit(X_train, Y_train)
	print("KNN predictions: ", knn._predict(X_test))

func _linreg_example():
	var X_train = mltools._dropVariable(dataLinR, dataLinR[0].size()-1)
	var Y_train = mltools._getVariable(dataLinR, dataLinR[0].size()-1)
	var X_test = [
		[7.2],
		[9.0],
		[11.1],
	]
	
	var linreg = DTDALinReg.new(0.01, 1000)
	linreg._fit(X_train, Y_train)
	print("Linear Regression predictions: ", linreg._predict(X_test))

func _logreg_example():
	var X_train = mltools._dropVariable(dataLogR, dataLogR[0].size()-1)
	var Y_train = mltools._getVariable(dataLogR, dataLogR[0].size()-1)
	var X_test = [
		[1, 3, 1, 0, 1, 0],
		[2, 2, 4, 1, 1, 1],
		[4, 1, 1, 0, 1, 0],
	]
	
	var logreg = DTDALogReg.new(0.01, 1000)
	logreg._fit(X_train, Y_train)
	print("Logistic Regression predictions: ", logreg._predict(X_test))
