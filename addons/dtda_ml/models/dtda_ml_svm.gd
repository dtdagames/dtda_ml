extends MLTools

class_name DTDASVM

# === SVM model === #
var m
var n
var lr
var lambda
var iter
var W
var b

func _init(learning_rate=0.01, lambda_param=0.01, n_iters=1000):
	lr = learning_rate
	lambda = lambda_param
	iter = n_iters

func _fit(X, Y):
	m = X.size()
	n = X[0].size()
	
	var y2 = _normalize_negative(Y)
	
	# list zeros
	W = []
	for i in n:
		W.push_back(0)
	b = 0
	
	# gradient
	for a in range(iter):
		for i in X.size():
			var dotXW = _dot_product_simple(X[i], W)
			var dotXWb = _sub_arrays_const(dotXW, b)
			var ti = _multiply_array_coef(dotXWb, y2[i])
			
			if ti[0] >= 1:
				var coefLW = _multiply_array_coef(W, 2*lambda)
				var coefLR = _multiply_array_coef(coefLW, lr)
				W = _substract_arrays(W, coefLR)
			else:
				var coefXY = _multiply_array_coef(X[i], y2[i])
				var coefLW = _multiply_array_coef(W, 2*lambda)
				var subLWXY = _substract_arrays(coefLW, coefXY)
				var coefLR = _multiply_array_coef(subLWXY, lr)
				W = _substract_arrays(W, coefLR)
				b = b - (lr*y2[i])

func _predict(newX):
	var dotXW = _dot_product(newX, W)
	var predY = _sub_arrays_const(dotXW, b)
	return _sign_array(predY)

# === End SVM model === #
