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

func _init(newRate:float, newIterations:int):
	rate = newRate
	iterations = newIterations

func _update_weights():
	var Y_pred = _predict(X)
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
	
	W = _array_zeros(n)
	b = 0
	X = newX
	Y = newY
	
	for i in iterations:
		_update_weights()

func _predict(newX):
	var dotXW = _dot_product(newX, W)
	return _add_arrays_const(dotXW, b)

# === End Linear Regression model === #
