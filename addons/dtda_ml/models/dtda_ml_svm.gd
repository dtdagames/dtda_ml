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
var scaler

func _init(learning_rate=0.01, lambda_param=0.01, n_iters=1000):
	lr = learning_rate
	lambda = lambda_param
	iter = n_iters

func _fit(newX, newY):
	m = newX.size()
	n = newX[0].size()

	# standardized features, the descent is unstable otherwise
	scaler = DTDAScaler.new()
	var X = scaler._fit_transform(newX)
	var y2 = _normalize_negative(newY)

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
	if not _check_fitted("DTDASVM", W):
		return []
	var dotXW = _dot_product(scaler._transform(newX), W)
	var predY = _sub_arrays_const(dotXW, b)
	return _sign_array(predY)

func _to_dict():
	if not _check_fitted("DTDASVM", W, "_save()"):
		return {}
	return {
		"model": "DTDASVM",
		"version": 1,
		"lr": lr,
		"lambda": lambda,
		"iter": iter,
		"W": W,
		"b": b,
		"scaler": scaler._to_dict(),
	}

func _from_dict(data):
	if not _check_model_name(data, "DTDASVM"):
		return false
	lr = data.get("lr", lr)
	lambda = data.get("lambda", lambda)
	iter = data.get("iter", iter)
	W = data.get("W")
	b = data.get("b", 0)
	if W == null:
		push_error("DTDASVM: the saved model has no weights")
		return false
	n = W.size()
	scaler = DTDAScaler.new()
	return scaler._from_dict(data.get("scaler", {}))

# === End SVM model === #
