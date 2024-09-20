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
	m = newX.size()
	n = newX[0].size()
	W = _array_zeros(n)
	b = 0
	X = newX
	Y = newY
	
	for i in iterations:
		_update_weights(i)

func _sigmoid(newX, newW, newB):
	# 1/(1 + e(-(x.dot(w) + b)
	var dotXW = _dot_product(newX, newW)
	var dotXWb = _add_arrays_const(dotXW, b)
	var dotXWbn = _multiply_array_coef(dotXWb, -1)
	var expXWb = _exp_array_(dotXWbn)
	var expXWb1 = _add_arrays_const(expXWb, 1)
	return _divide_inverse_array_coef(expXWb1, 1)

func _predict(newX):
	var Z = _sigmoid(newX, W, b)
	var matrix = []
	for i in Z:
		matrix.push_back(round(i))
	return matrix

# === End Logistic Regression model === #
