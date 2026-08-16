class_name MLTools

# shared check for every function comparing predictions to expected labels
func _check_pair(caller, y_pred, y_test):
	if y_pred.size() == 0:
		push_error("MLTools: %s called without any prediction" % caller)
		return false
	if y_pred.size() != y_test.size():
		push_error("MLTools: %s got %d predictions for %d expected labels" % [caller, y_pred.size(), y_test.size()])
		return false
	return true

# type : 0 KNN, 1 linear reg, 2 logistic reg, 3 SVM
func _get_perf(y_pred, y_test, type):
	if not _check_pair("_get_perf()", y_pred, y_test):
		return 0.0

	# convert >0.5 to 1 from prediction for linear regression
	if type == 1:
		y_pred = _normalize_int(y_pred)
	# convert 0 to -1 from test for SVM
	if type == 3:
		y_pred = _normalize_negative(y_pred)
		y_test = _normalize_negative(y_test)
	
	var correctly_classified = 0
	var count = 0
	for i in y_pred.size() :
		if y_test[i] == y_pred[i]:
			correctly_classified += 1
		count += 1
	
	return snapped(float(correctly_classified) / float(count) *100, 0.01)

# convert array from float to int
func array_to_int(arr):
	var tempData = []
	for i in arr:
		tempData.push_back(int(i))
	return tempData

# convert 0 to -1 from array
func _normalize_negative(tempData):
	var newData = []
	for row in tempData:
		if row == 0:
			row = -1
		newData.push_back(row)
	return newData
# convert >0.5 to 1 from array
func _normalize_int(tempData):
	var newData = []
	for row in tempData:
		if row > 0.5:
			row = 1
		else:
			row = 0
		newData.push_back(row)
	return newData
# convert value to -1/1 from array, a value sitting exactly on the boundary goes to 1
func _sign_array(x):
	var matrix = []
	for row in x:
		if row>=0:
			row = 1
		else:
			row = -1
		matrix.push_back(row)
	return matrix

# return array with specific column
func _getVariable(tempData, tempColumnId):
	var newData = []
	for row in tempData:
		newData.push_back(row[tempColumnId])
	return newData

# return array without specific column
func _dropVariable(tempData, tempColumnId):
	var newData = []
	for i in tempData.size():
		newData.push_back([])
		for u in tempData[i].size():
			if u != tempColumnId:
				newData[i].push_back(tempData[i][u])
	return newData

# return array of zeros
func _array_zeros(n):
	var tempW = []
	for i in n:
		tempW.push_back(0)
	return tempW

# return substract of two arrays
func _substract_arrays(x1, x2):
	var matrix = []
	for i in x1.size():
		matrix.push_back(x1[i] - x2[i])
	return matrix
# return substract of array and const
func _sub_arrays_const(x1, b):
	var matrix = []
	for i in x1.size():
		matrix.push_back(x1[i] - b)
	return matrix

# add array by constant
func _add_arrays_const(x1, b):
	var matrix = []
	for i in x1.size():
		matrix.push_back(x1[i] + b)
	return matrix

# mutliply rows of array by coef
func _multiply_array_coef(x1, b):
	var matrix = []
	for i in x1.size():
		matrix.push_back(x1[i] * b)
	return matrix
# divide rows of array by coef
func _divide_array_coef(x1, b):
	var matrix = []
	for i in x1.size():
		matrix.push_back(x1[i] / b)
	return matrix

# divide coef by rows 
func _divide_inverse_array_coef(x1, b):
	var matrix = []
	for i in x1.size():
		matrix.push_back(b / x1[i])
	return matrix

# return rows of array by exp
func _exp_array_(x1):
	var matrix = []
	for i in x1.size():
		matrix.push_back(exp(x1[i]))
	return matrix

# return dot product of two arrays
func _dot_product(x1, x2):
	var matrix = []
	for i in x1.size():
		var res = 0
		for u in x1[0].size():
			res += x1[i][u] * x2[u]
		matrix.push_back(res)
	return matrix
# return dor product of array and const
func _dot_product_simple(x1, x2):
	var matrix = []
	var res = 0
	for i in x1.size():
		res += x1[i] * x2[i]
	matrix.push_back(res)
	return matrix

# transpose array
func _transpose_array(x):
	var matrix = []
	for i in x[0].size():
		matrix.push_back([])
		for u in x.size():
			matrix[i].push_back(x[u][i])
	return matrix
# transpose 1D array
func _transpose_simple_array(x):
	var matrix = []
	for i in x.size():
		matrix.push_back(x[i])
	return matrix

# return sum of all rows
func _sum_array(x):
	var total = 0
	for i in x.size():
		total += x[i]
	return total

# return mean of all rows
func _mean_array(x):
	if x.size() == 0:
		return 0.0
	return float(_sum_array(x)) / float(x.size())

# return standard deviation of all rows, 1.0 when constant so it stays safe to divide by
func _std_array(x):
	if x.size() == 0:
		return 1.0
	var mean = _mean_array(x)
	var total = 0.0
	for i in x.size():
		total += (x[i] - mean)**2
	var deviation = sqrt(total / float(x.size()))
	if deviation == 0.0:
		return 1.0
	return deviation

# report a clear error instead of crashing deep in the math
func _check_fitted(model_name, trained_value, called = "_predict()"):
	if trained_value == null:
		push_error("%s: %s called before _fit()" % [model_name, called])
		return false
	return true

# wrap a 1D array into a single column matrix, so DTDAScaler can handle a target
func _column_to_matrix(x):
	var matrix = []
	for i in x.size():
		matrix.push_back([x[i]])
	return matrix

# unwrap a single column matrix back into a 1D array
func _matrix_to_column(x):
	var column = []
	for i in x.size():
		column.push_back(x[i][0])
	return column

# === Classification metrics === #

# percentage of correct answers
func _accuracy(y_pred, y_test):
	if not _check_pair("_accuracy()", y_pred, y_test):
		return 0.0
	var correct = 0
	for i in y_pred.size():
		if y_pred[i] == y_test[i]:
			correct += 1
	return snapped(float(correct) / float(y_pred.size()) * 100, 0.01)

# true/false positives and negatives around a given positive label
func _confusion_matrix(y_pred, y_test, positive = 1):
	if not _check_pair("_confusion_matrix()", y_pred, y_test):
		return {}
	var counts = {"tp": 0, "fp": 0, "tn": 0, "fn": 0}
	for i in y_pred.size():
		var predicted_positive = y_pred[i] == positive
		var actually_positive = y_test[i] == positive
		if predicted_positive and actually_positive:
			counts["tp"] += 1
		elif predicted_positive:
			counts["fp"] += 1
		elif actually_positive:
			counts["fn"] += 1
		else:
			counts["tn"] += 1
	return counts

# share of the predicted positives that are right, from 0 to 1
func _precision(y_pred, y_test, positive = 1):
	var counts = _confusion_matrix(y_pred, y_test, positive)
	if counts.is_empty():
		return 0.0
	var predicted = counts["tp"] + counts["fp"]
	# nothing was predicted positive, so nothing was predicted wrong either
	if predicted == 0:
		return 0.0
	return snapped(float(counts["tp"]) / float(predicted), 0.0001)

# share of the real positives that were found, from 0 to 1
func _recall(y_pred, y_test, positive = 1):
	var counts = _confusion_matrix(y_pred, y_test, positive)
	if counts.is_empty():
		return 0.0
	var actual = counts["tp"] + counts["fn"]
	if actual == 0:
		return 0.0
	return snapped(float(counts["tp"]) / float(actual), 0.0001)

# harmonic mean of precision and recall, from 0 to 1
func _f1_score(y_pred, y_test, positive = 1):
	var p = _precision(y_pred, y_test, positive)
	var r = _recall(y_pred, y_test, positive)
	if p + r == 0:
		return 0.0
	return snapped(2 * p * r / (p + r), 0.0001)

# === Regression metrics === #

# mean squared error
func _mse(y_pred, y_test):
	if not _check_pair("_mse()", y_pred, y_test):
		return 0.0
	var total = 0.0
	for i in y_pred.size():
		total += (y_test[i] - y_pred[i])**2
	return total / float(y_pred.size())

# root mean squared error, in the unit of the target
func _rmse(y_pred, y_test):
	return sqrt(_mse(y_pred, y_test))

# mean absolute error, less sensitive to outliers than the RMSE
func _mae(y_pred, y_test):
	if not _check_pair("_mae()", y_pred, y_test):
		return 0.0
	var total = 0.0
	for i in y_pred.size():
		total += abs(y_test[i] - y_pred[i])
	return total / float(y_pred.size())

# share of the variance explained by the model, 1.0 is a perfect fit
# a model worse than always answering the mean scores below 0
func _r2_score(y_pred, y_test):
	if not _check_pair("_r2_score()", y_pred, y_test):
		return 0.0
	var mean = _mean_array(y_test)
	var residual = 0.0
	var total = 0.0
	for i in y_test.size():
		residual += (y_test[i] - y_pred[i])**2
		total += (y_test[i] - mean)**2
	# every expected value is the same, the variance to explain is null
	if total == 0.0:
		return 0.0
	return snapped(1.0 - residual / total, 0.0001)

# === Saving and loading === #

# overridden by every model
func _to_dict():
	push_error("MLTools: this class cannot be saved")
	return {}

func _from_dict(_data):
	push_error("MLTools: this class cannot be loaded")
	return false

# guard against loading a KNN file into a linear regression
func _check_model_name(data, expected):
	var found = data.get("model", "unknown")
	if found != expected:
		push_error("%s: this file holds a '%s' model" % [expected, found])
		return false
	return true

# write a trained model to a JSON file, returns true on success
# use a user:// path, res:// is read only once the game is exported
func _save(path):
	var data = _to_dict()
	if data.is_empty():
		return false
	var file = FileAccess.open(path, FileAccess.WRITE)
	if file == null:
		push_error("MLTools: cannot write %s (%s)" % [path, error_string(FileAccess.get_open_error())])
		return false
	# full precision, otherwise the weights are truncated on the way out
	file.store_string(JSON.stringify(data, "\t", true, true))
	file.close()
	return true

# read a model back from a JSON file, returns true on success
func _load(path):
	if not FileAccess.file_exists(path):
		push_error("MLTools: %s does not exist" % path)
		return false
	var file = FileAccess.open(path, FileAccess.READ)
	if file == null:
		push_error("MLTools: cannot read %s (%s)" % [path, error_string(FileAccess.get_open_error())])
		return false
	var data = JSON.parse_string(file.get_as_text())
	file.close()
	if typeof(data) != TYPE_DICTIONARY:
		push_error("MLTools: %s is not a valid model file" % path)
		return false
	return _from_dict(data)
