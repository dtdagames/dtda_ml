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
# scaling of the target, the features are scaled by MLTools
var Y_mean
var Y_std

func _init(newRate:float, newIterations:int):
	rate = newRate
	iterations = newIterations

# center and reduce the target
func _scale_target(newY):
	var matrix = []
	for i in newY.size():
		matrix.push_back((newY[i] - Y_mean) / Y_std)
	return matrix

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
	
	_fit_feature_scaling(newX)
	Y_mean = _mean_array(newY)
	Y_std = _std_array(newY)

	W = _array_zeros(n)
	b = 0
	X = _scale_features(newX)
	Y = _scale_target(newY)

	for i in iterations:
		_update_weights()

func _predict(newX):
	if not _check_fitted("DTDALinReg", W):
		return []
	var pred = _predict_scaled(_scale_features(newX))
	# back to the unit of the training target
	return _add_arrays_const(_multiply_array_coef(pred, Y_std), Y_mean)

# === End Linear Regression model === #
