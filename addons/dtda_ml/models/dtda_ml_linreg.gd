extends MLTools

class_name DTDALinReg

# === Linear Regression model === #
var m
var n
var rate
var iterations
var W
var b
var X
var Y
# features and target are standardized before the descent
var x_scaler
var y_scaler

func _init(newRate:float, newIterations:int):
	rate = newRate
	iterations = newIterations

# prediction in the standardized space, used by the descent
func _predict_scaled(scaledX):
	var dotXW = _dot_product(scaledX, W)
	return _add_arrays_const(dotXW, b)

func _update_weights():
	var Y_pred = _predict_scaled(X)
	# gradients
	var subY = _substract_arrays(Y, Y_pred)
	var dotXY = _dot_product(_transpose_array(X), subY)
	var dotXY2 = _multiply_array_coef(dotXY, -2)
	var dW = _divide_array_coef(dotXY2, m)
	var sumY = _sum_array(subY)
	var dB =  (-2 * sumY) / m
	# update
	var dWrate = _multiply_array_coef(dW, rate)
	W = _substract_arrays(W, dWrate)
	b = b - (rate * dB)

func _fit(newX, newY):
	# The rows are weighed before a single field is written: a fit that took them as
	# they came would leave a working model holding a nan, or half rewritten by a
	# raise in the middle. Answers false when it refuses, true when it fitted
	if not _check_matrix(newX, "DTDALinReg"):
		return false
	if not _check_labels(newX, newY, "DTDALinReg"):
		return false
	# this one scales the target and descends on it, so a label has to be a number
	if not _check_number_array(newY, "DTDALinReg", "labels"):
		return false
	m = newX.size()
	n = newX[0].size()
	
	x_scaler = DTDAScaler.new()
	y_scaler = DTDAScaler.new()

	W = _array_zeros(n)
	b = 0
	X = x_scaler._fit_transform(newX)
	Y = _matrix_to_column(y_scaler._fit_transform(_column_to_matrix(newY)))

	for i in iterations:
		_update_weights()
	return true

func _predict(newX):
	if not _check_fitted("DTDALinReg", W):
		return []
	var pred = _predict_scaled(x_scaler._transform(newX))
	# back to the unit of the training target
	return _matrix_to_column(y_scaler._inverse_transform(_column_to_matrix(pred)))

func _to_dict():
	if not _check_fitted("DTDALinReg", W, "_save()"):
		return {}
	return {
		"model": "DTDALinReg",
		"version": 1,
		"rate": rate,
		"iterations": iterations,
		"W": W,
		"b": b,
		"x_scaler": x_scaler._to_dict(),
		"y_scaler": y_scaler._to_dict(),
	}

func _from_dict(data):
	if not _check_model_name(data, "DTDALinReg"):
		return false
	# Everything is read aside first and only takes the place of the standing model
	# once the whole file is known to be readable. Assigning as it went used to leave
	# a working model with the weights of a file it had just refused
	# absent, malformed and unusable answer the same way: a null is not a list either,
	# and _predict() computes with every one of these numbers
	var saved_W = data.get("W")
	if not _check_number_array(saved_W, "DTDALinReg", "weights"):
		return false
	var saved_b = data.get("b", 0)
	if not _check_number(saved_b, "DTDALinReg", "intercept"):
		return false
	var saved_x = DTDAScaler.new()
	var saved_y = DTDAScaler.new()
	if not saved_x._from_dict(data.get("x_scaler", {})):
		return false
	if not saved_y._from_dict(data.get("y_scaler", {})):
		return false
	rate = data.get("rate", rate)
	iterations = data.get("iterations", iterations)
	W = saved_W
	b = saved_b
	n = W.size()
	x_scaler = saved_x
	y_scaler = saved_y
	return true

# === End Linear Regression model === #
