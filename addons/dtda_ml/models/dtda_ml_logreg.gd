extends MLTools

class_name DTDALogReg

# === Logistic Regression model === #
var m
var n
var rate
var iterations
var W
var b
var X
var Y
var scaler

func _init(newRate:float, newIterations:int):
	rate = newRate
	iterations = newIterations

func _update_weights(i):
	var a = _sigmoid(X, W, b)
	
	# gradients
	var tmp = _substract_arrays(a, _transpose_simple_array(Y))
	var dotW = _dot_product(_transpose_array(X), tmp)
	var dW = _divide_array_coef(dotW, m)
	var dB = _sum_array(tmp) / m
	
	# update
	var dWrate = _multiply_array_coef(dW, rate)
	W = _substract_arrays(W, dWrate)
	b = b - (rate * dB)

func _fit(newX, newY):
	# The rows are weighed before a single field is written: a fit that took them as
	# they came would leave a working model holding a nan, or half rewritten by a
	# raise in the middle. Answers false when it refuses, true when it fitted
	if not _check_matrix(newX, "DTDALogReg"):
		return false
	if not _check_labels(newX, newY, "DTDALogReg"):
		return false
	# the descent computes with the label itself, so it has to be a number
	if not _check_number_array(newY, "DTDALogReg", "labels"):
		return false
	m = newX.size()
	n = newX[0].size()
	W = _array_zeros(n)
	b = 0
	# standardized features, otherwise exp() overflows as soon as the data gets large
	scaler = DTDAScaler.new()
	X = scaler._fit_transform(newX)
	Y = newY

	for i in iterations:
		_update_weights(i)
	return true

func _sigmoid(newX, newW, newB):
	# 1/(1 + e(-(x.dot(w) + b)
	var dotXW = _dot_product(newX, newW)
	var dotXWb = _add_arrays_const(dotXW, newB)
	var dotXWbn = _multiply_array_coef(dotXWb, -1)
	var expXWb = _exp_array_(dotXWbn)
	var expXWb1 = _add_arrays_const(expXWb, 1)
	return _divide_inverse_array_coef(expXWb1, 1)

func _predict(newX):
	if not _check_fitted("DTDALogReg", W):
		return []
	var Z = _sigmoid(scaler._transform(newX), W, b)
	var matrix = []
	for i in Z:
		matrix.push_back(round(i))
	return matrix

func _to_dict():
	if not _check_fitted("DTDALogReg", W, "_save()"):
		return {}
	return {
		"model": "DTDALogReg",
		"version": 1,
		"rate": rate,
		"iterations": iterations,
		"W": W,
		"b": b,
		"scaler": scaler._to_dict(),
	}

func _from_dict(data):
	if not _check_model_name(data, "DTDALogReg"):
		return false
	# Everything is read aside first and only takes the place of the standing model
	# once the whole file is known to be readable. Assigning as it went used to leave
	# a working model with the weights of a file it had just refused
	# absent, malformed and unusable answer the same way: a null is not a list either,
	# and _predict() computes with every one of these numbers
	var saved_W = data.get("W")
	if not _check_number_array(saved_W, "DTDALogReg", "weights"):
		return false
	var saved_b = data.get("b", 0)
	if not _check_number(saved_b, "DTDALogReg", "intercept"):
		return false
	var saved_scaler = DTDAScaler.new()
	if not saved_scaler._from_dict(data.get("scaler", {})):
		return false
	rate = data.get("rate", rate)
	iterations = data.get("iterations", iterations)
	W = saved_W
	b = saved_b
	n = W.size()
	scaler = saved_scaler
	return true

# === End Logistic Regression model === #
