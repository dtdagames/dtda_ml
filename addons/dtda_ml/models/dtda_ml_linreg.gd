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
	rate = data.get("rate", rate)
	iterations = data.get("iterations", iterations)
	W = data.get("W")
	b = data.get("b", 0)
	if W == null:
		push_error("DTDALinReg: the saved model has no weights")
		return false
	n = W.size()
	x_scaler = DTDAScaler.new()
	y_scaler = DTDAScaler.new()
	return x_scaler._from_dict(data.get("x_scaler", {})) and y_scaler._from_dict(data.get("y_scaler", {}))

# === End Linear Regression model === #
