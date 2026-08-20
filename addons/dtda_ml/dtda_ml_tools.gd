class_name DTDATools

# shared check for every function comparing predictions to expected labels
func _check_pair(caller: String, y_pred, y_test) -> bool:
	if y_pred.size() == 0:
		push_error("DTDATools: %s called without any prediction" % caller)
		return false
	if y_pred.size() != y_test.size():
		push_error("DTDATools: %s got %d predictions for %d expected labels" % [caller, y_pred.size(), y_test.size()])
		return false
	return true

# type : 0 KNN, 1 linear reg, 2 logistic reg, 3 SVM
func get_perf(y_pred, y_test, type: int) -> float:
	if not _check_pair("get_perf()", y_pred, y_test):
		return 0.0

	# convert >0.5 to 1 from prediction for linear regression
	if type == 1:
		y_pred = _normalize_int(y_pred)
	# convert 0 to -1 from test for SVM
	if type == 3:
		y_pred = _normalize_negative(y_pred)
		y_test = _normalize_negative(y_test)
	
	var correctly_classified: int = 0
	var count: int = 0
	for i in y_pred.size() :
		if y_test[i] == y_pred[i]:
			correctly_classified += 1
		count += 1
	
	return snapped(float(correctly_classified) / float(count) *100, 0.01)

# convert array from float to int
func array_to_int(arr) -> Array:
	var tempData = []
	for i in arr:
		tempData.push_back(int(i))
	return tempData

# convert 0 to -1 from array
func _normalize_negative(tempData) -> Array:
	var newData = []
	for row in tempData:
		if row == 0:
			row = -1
		newData.push_back(row)
	return newData
# convert >0.5 to 1 from array
func _normalize_int(tempData) -> Array:
	var newData = []
	for row in tempData:
		if row > 0.5:
			row = 1
		else:
			row = 0
		newData.push_back(row)
	return newData
# convert value to -1/1 from array, a value sitting exactly on the boundary goes to 1
func _sign_array(x) -> Array:
	var matrix = []
	for row in x:
		if row>=0:
			row = 1
		else:
			row = -1
		matrix.push_back(row)
	return matrix

# return array with specific column
func get_variable(tempData, tempColumnId: int) -> Array:
	var newData = []
	for row in tempData:
		newData.push_back(row[tempColumnId])
	return newData

# return array without specific column
func drop_variable(tempData, tempColumnId: int) -> Array:
	var newData = []
	for i in tempData.size():
		newData.push_back([])
		for u in tempData[i].size():
			if u != tempColumnId:
				newData[i].push_back(tempData[i][u])
	return newData

# return array of zeros
func _array_zeros(n: int) -> Array:
	var tempW: Array = []
	for i in n:
		tempW.push_back(0)
	return tempW

# return substract of two arrays
func _substract_arrays(x1, x2) -> Array:
	var matrix: Array = []
	for i in x1.size():
		matrix.push_back(x1[i] - x2[i])
	return matrix
# return substract of array and const
func _sub_arrays_const(x1, b: float) -> Array:
	var matrix: Array = []
	for i in x1.size():
		matrix.push_back(x1[i] - b)
	return matrix

# add array by constant
func _add_arrays_const(x1, b: float) -> Array:
	var matrix: Array = []
	for i in x1.size():
		matrix.push_back(x1[i] + b)
	return matrix

# mutliply rows of array by coef
func _multiply_array_coef(x1, b: float) -> Array:
	var matrix: Array = []
	for i in x1.size():
		matrix.push_back(x1[i] * b)
	return matrix
# divide rows of array by coef
# float() so a division between two integers does not discard the decimal part
func _divide_array_coef(x1, b: float) -> Array:
	var matrix: Array = []
	for i in x1.size():
		matrix.push_back(x1[i] / float(b))
	return matrix

# divide coef by rows
func _divide_inverse_array_coef(x1, b: float) -> Array:
	var matrix: Array = []
	for i in x1.size():
		matrix.push_back(float(b) / x1[i])
	return matrix

# return rows of array by exp
func _exp_array_(x1) -> Array:
	var matrix: Array = []
	for i in x1.size():
		matrix.push_back(exp(x1[i]))
	return matrix

# return dot product of two arrays
func _dot_product(x1, x2) -> Array:
	var matrix: Array = []
	for i in x1.size():
		var res: float = 0.0
		for u in x1[0].size():
			res += x1[i][u] * x2[u]
		matrix.push_back(res)
	return matrix
# return dor product of array and const
func _dot_product_simple(x1, x2) -> Array:
	var matrix: Array = []
	var res: float = 0.0
	for i in x1.size():
		res += x1[i] * x2[i]
	matrix.push_back(res)
	return matrix

# transpose array
func _transpose_array(x) -> Array:
	var matrix: Array = []
	for i in x[0].size():
		matrix.push_back([])
		for u in x.size():
			matrix[i].push_back(x[u][i])
	return matrix
# transpose 1D array
func _transpose_simple_array(x) -> Array:
	var matrix: Array = []
	for i in x.size():
		matrix.push_back(x[i])
	return matrix

# return sum of all rows
func _sum_array(x) -> float:
	var total: float = 0.0
	for i in x.size():
		total += x[i]
	return total

# return mean of all rows
func _mean_array(x) -> float:
	if x.size() == 0:
		return 0.0
	return float(_sum_array(x)) / float(x.size())

# return standard deviation of all rows, 1.0 when constant so it stays safe to divide by
func _std_array(x) -> float:
	if x.size() == 0:
		return 1.0
	var mean: float = _mean_array(x)
	var total: float = 0.0
	for i in x.size():
		total += (x[i] - mean)**2
	var deviation: float = sqrt(total / float(x.size()))
	if deviation == 0.0:
		return 1.0
	return deviation

# report a clear error instead of crashing deep in the math
func _check_fitted(model_name: String, trained_value, called: String = "predict()") -> bool:
	if trained_value == null:
		push_error("%s: %s called before fit()" % [model_name, called])
		return false
	return true

# wrap a 1D array into a single column matrix, so DTDAScaler can handle a target
func _column_to_matrix(x) -> Array:
	var matrix: Array = []
	for i in x.size():
		matrix.push_back([x[i]])
	return matrix

# unwrap a single column matrix back into a 1D array
func _matrix_to_column(x) -> Array:
	var column: Array = []
	for i in x.size():
		column.push_back(x[i][0])
	return column

# === Classification metrics === #

# percentage of correct answers
func accuracy(y_pred, y_test) -> float:
	if not _check_pair("accuracy()", y_pred, y_test):
		return 0.0
	var correct: int = 0
	for i in y_pred.size():
		if y_pred[i] == y_test[i]:
			correct += 1
	return snapped(float(correct) / float(y_pred.size()) * 100, 0.01)

# true/false positives and negatives around a given positive label
func confusion_matrix(y_pred, y_test, positive = 1) -> Dictionary:
	if not _check_pair("confusion_matrix()", y_pred, y_test):
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
func precision(y_pred, y_test, positive = 1) -> float:
	var counts = confusion_matrix(y_pred, y_test, positive)
	if counts.is_empty():
		return 0.0
	var predicted = counts["tp"] + counts["fp"]
	# nothing was predicted positive, so nothing was predicted wrong either
	if predicted == 0:
		return 0.0
	return snapped(float(counts["tp"]) / float(predicted), 0.0001)

# share of the real positives that were found, from 0 to 1
func recall(y_pred, y_test, positive = 1) -> float:
	var counts = confusion_matrix(y_pred, y_test, positive)
	if counts.is_empty():
		return 0.0
	var actual = counts["tp"] + counts["fn"]
	if actual == 0:
		return 0.0
	return snapped(float(counts["tp"]) / float(actual), 0.0001)

# harmonic mean of precision and recall, from 0 to 1
func f1_score(y_pred, y_test, positive = 1) -> float:
	var p = precision(y_pred, y_test, positive)
	var r = recall(y_pred, y_test, positive)
	if p + r == 0:
		return 0.0
	return snapped(2 * p * r / (p + r), 0.0001)

# === Regression metrics === #

# mean squared error
func mse(y_pred, y_test) -> float:
	if not _check_pair("mse()", y_pred, y_test):
		return 0.0
	var total: float = 0.0
	for i in y_pred.size():
		total += (y_test[i] - y_pred[i])**2
	return total / float(y_pred.size())

# root mean squared error, in the unit of the target
func rmse(y_pred, y_test) -> float:
	return sqrt(mse(y_pred, y_test))

# mean absolute error, less sensitive to outliers than the RMSE
func mae(y_pred, y_test) -> float:
	if not _check_pair("mae()", y_pred, y_test):
		return 0.0
	var total: float = 0.0
	for i in y_pred.size():
		total += abs(y_test[i] - y_pred[i])
	return total / float(y_pred.size())

# share of the variance explained by the model, 1.0 is a perfect fit
# a model worse than always answering the mean scores below 0
func r2_score(y_pred, y_test) -> float:
	if not _check_pair("r2_score()", y_pred, y_test):
		return 0.0
	var mean: float = _mean_array(y_test)
	var residual: float = 0.0
	var total: float = 0.0
	for i in y_test.size():
		residual += (y_test[i] - y_pred[i])**2
		total += (y_test[i] - mean)**2
	# every expected value is the same, the variance to explain is null
	if total == 0.0:
		return 0.0
	return snapped(1.0 - residual / total, 0.0001)

# === Saving and loading === #

# overridden by every model
func to_dict() -> Dictionary:
	push_error("DTDATools: this class cannot be saved")
	return {}

func from_dict(_data) -> bool:
	push_error("DTDATools: this class cannot be loaded")
	return false

# A number a model read out of a file has to be one, not merely present. A file lives
# in user://, where it can be edited by hand, and a text where a number belongs would
# load and only fall apart at the first prediction
func _check_number(value, model_name: String, field: String) -> bool:
	if not (typeof(value) in [TYPE_INT, TYPE_FLOAT]):
		push_error("%s: the saved %s is not a number" % [model_name, field])
		return false
	# a nan and an inf carry a numeric type and are not numbers anything can compute
	# with: a nan answers false to every comparison and spreads through every weight
	# it touches without a word
	if typeof(value) == TYPE_FLOAT and not is_finite(value):
		push_error("%s: the saved %s is %s, which is not a number to compute with" % [model_name, field, value])
		return false
	return true

# the same for a list of numbers, which also has to hold something
func _check_number_array(values, model_name: String, field: String) -> bool:
	if typeof(values) != TYPE_ARRAY:
		push_error("%s: the saved %s is not a list" % [model_name, field])
		return false
	if values.size() == 0:
		push_error("%s: the saved %s is empty" % [model_name, field])
		return false
	for value in values:
		if not (typeof(value) in [TYPE_INT, TYPE_FLOAT]):
			push_error("%s: the saved %s holds something that is not a number" % [model_name, field])
			return false
		if typeof(value) == TYPE_FLOAT and not is_finite(value):
			push_error("%s: the saved %s holds %s, which is not a number to compute with" % [model_name, field, value])
			return false
	return true

# What fit() is handed has to be something it can compute with, and it arrives from
# the caller rather than from a file: one unlucky division upstream is enough. A model
# that was working must not be left holding a nan, or half rewritten by a fit that
# raised in the middle, so the rows are weighed before anything is written down
func _check_matrix(X, model_name: String) -> bool:
	if typeof(X) != TYPE_ARRAY or X.size() == 0:
		push_error("%s: fit() got no rows to learn from" % model_name)
		return false
	var width: int = 0
	for i in X.size():
		if not _check_number_array(X[i], model_name, "row %d" % i):
			return false
		if i == 0:
			width = X[i].size()
		elif X[i].size() != width:
			push_error("%s: fit() got a row of %d columns next to a row of %d" % [model_name, X[i].size(), width])
			return false
	return true

# as many labels as there are rows. What the labels hold is left alone: a KNN answers
# them back as they came and a classifier only counts them, so a label can be a string
# and often is. The models that do arithmetic on a label weigh it themselves
func _check_labels(X, y, model_name: String) -> bool:
	if typeof(y) != TYPE_ARRAY:
		push_error("%s: fit() got labels that are not a list" % model_name)
		return false
	if y.size() != X.size():
		push_error("%s: fit() got %d rows and %d labels" % [model_name, X.size(), y.size()])
		return false
	return true

# guard against loading a KNN file into a linear regression
func _check_model_name(data, expected: String) -> bool:
	var found = data.get("model", "unknown")
	if found != expected:
		push_error("%s: this file holds a '%s' model" % [expected, found])
		return false
	return true

# write a trained model to a JSON file, returns true on success
# use a user:// path, res:// is read only once the game is exported
#
# No self. on the calls to this one, and that is not an oversight: there is no global
# save() for a bare save(path) to reach, so it finds this method. Its neighbour load()
# has a global of the same name, which wins, and every call to it is qualified for
# that reason alone. Do not even them up.
func save(path: String) -> bool:
	var data = to_dict()
	if data.is_empty():
		return false
	var file = FileAccess.open(path, FileAccess.WRITE)
	if file == null:
		push_error("DTDATools: cannot write %s (%s)" % [path, error_string(FileAccess.get_open_error())])
		return false
	# full precision, otherwise the weights are truncated on the way out
	file.store_string(JSON.stringify(data, "\t", true, true))
	file.close()
	return true

# read a model back from a JSON file, returns true on success
func load(path: String) -> bool:
	if not FileAccess.file_exists(path):
		push_error("DTDATools: %s does not exist" % path)
		return false
	var file = FileAccess.open(path, FileAccess.READ)
	if file == null:
		push_error("DTDATools: cannot read %s (%s)" % [path, error_string(FileAccess.get_open_error())])
		return false
	var data = JSON.parse_string(file.get_as_text())
	file.close()
	if typeof(data) != TYPE_DICTIONARY:
		push_error("DTDATools: %s is not a valid model file" % path)
		return false
	return from_dict(data)

# === The older names === #
# Every method above used to carry a leading underscore, which in Godot marks a
# method as virtual or private: the engine calls _ready() and _process(), you do not.
# The names below are the ones that shipped, kept working so nothing that already
# calls them breaks. They only forward. Prefer the ones without the underscore.

func _get_perf(y_pred, y_test, type):
	return get_perf(y_pred, y_test, type)

func _getVariable(tempData, tempColumnId):
	return get_variable(tempData, tempColumnId)

func _dropVariable(tempData, tempColumnId):
	return drop_variable(tempData, tempColumnId)

func _accuracy(y_pred, y_test):
	return accuracy(y_pred, y_test)

func _confusion_matrix(y_pred, y_test, positive = 1):
	return confusion_matrix(y_pred, y_test, positive)

func _precision(y_pred, y_test, positive = 1):
	return precision(y_pred, y_test, positive)

func _recall(y_pred, y_test, positive = 1):
	return recall(y_pred, y_test, positive)

func _f1_score(y_pred, y_test, positive = 1):
	return f1_score(y_pred, y_test, positive)

func _mse(y_pred, y_test):
	return mse(y_pred, y_test)

func _rmse(y_pred, y_test):
	return rmse(y_pred, y_test)

func _mae(y_pred, y_test):
	return mae(y_pred, y_test)

func _r2_score(y_pred, y_test):
	return r2_score(y_pred, y_test)

func _to_dict():
	return to_dict()

func _from_dict(_data):
	return from_dict(_data)

func _save(path):
	return save(path)

# self. is not decoration here: load() on its own is the engine global that reads
# a resource, and it wins over a method of the same name inside the class. From
# outside, model.load(path) reaches this one, the way ConfigFile.load() does
func _load(path):
	return self.load(path)
