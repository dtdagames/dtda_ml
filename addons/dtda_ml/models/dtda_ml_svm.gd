extends DTDATools

class_name DTDASVM

# === SVM model === #
var m: int = 0
var n: int = 0
var lr: float
var lambda: float
var iter: int
var W
var b
var scaler

func _init(learning_rate: float = 0.01, lambda_param: float = 0.01, n_iters: int = 1000) -> void:
	lr = learning_rate
	lambda = lambda_param
	iter = n_iters

func fit(newX, newY) -> bool:
	# The rows are weighed before a single field is written: a fit that took them as
	# they came would leave a working model holding a nan, or half rewritten by a
	# raise in the middle. Answers false when it refuses, true when it fitted
	if not _check_matrix(newX, "DTDASVM"):
		return false
	if not _check_labels(newX, newY, "DTDASVM"):
		return false
	# the descent multiplies by the label, so it has to be a number
	if not _check_number_array(newY, "DTDASVM", "labels"):
		return false
	m = newX.size()
	n = newX[0].size()

	# standardized features, the descent is unstable otherwise
	scaler = DTDAScaler.new()
	var X: Array = scaler.fit_transform(newX)
	var y2: Array = _normalize_negative(newY)

	# list zeros
	W = []
	for i in n:
		W.push_back(0)
	b = 0

	# gradient
	for a in range(iter):
		for i in X.size():
			var dotXW: Array = _dot_product_simple(X[i], W)
			var dotXWb: Array = _sub_arrays_const(dotXW, b)
			var ti: Array = _multiply_array_coef(dotXWb, y2[i])
			
			if ti[0] >= 1:
				var coefLW: Array = _multiply_array_coef(W, 2.0 * lambda)
				var coefLR: Array = _multiply_array_coef(coefLW, lr)
				W = _substract_arrays(W, coefLR)
			else:
				var coefXY: Array = _multiply_array_coef(X[i], y2[i])
				var coefLW: Array = _multiply_array_coef(W, 2.0 * lambda)
				var subLWXY: Array = _substract_arrays(coefLW, coefXY)
				var coefLR: Array = _multiply_array_coef(subLWXY, lr)
				W = _substract_arrays(W, coefLR)
				b = b - (lr*y2[i])
	return true

func predict(newX) -> Array:
	if not _check_fitted("DTDASVM", W):
		return []
	var dotXW = _dot_product(scaler.transform(newX), W)
	var predY = _sub_arrays_const(dotXW, b)
	return _sign_array(predY)

func to_dict():
	if not _check_fitted("DTDASVM", W, "save()"):
		return {}
	return {
		"model": "DTDASVM",
		"version": 1,
		"lr": lr,
		"lambda": lambda,
		"iter": iter,
		"W": W,
		"b": b,
		"scaler": scaler.to_dict(),
	}

func from_dict(data):
	if not _check_model_name(data, "DTDASVM"):
		return false
	# Everything is read aside first and only takes the place of the standing model
	# once the whole file is known to be readable. Assigning as it went used to leave
	# a working model with the weights of a file it had just refused
	# absent, malformed and unusable answer the same way: a null is not a list either,
	# and predict() computes with every one of these numbers
	var saved_W = data.get("W")
	if not _check_number_array(saved_W, "DTDASVM", "weights"):
		return false
	var saved_b = data.get("b", 0)
	if not _check_number(saved_b, "DTDASVM", "intercept"):
		return false
	var saved_scaler = DTDAScaler.new()
	if not saved_scaler.from_dict(data.get("scaler", {})):
		return false
	lr = data.get("lr", lr)
	lambda = data.get("lambda", lambda)
	iter = data.get("iter", iter)
	W = saved_W
	b = saved_b
	n = W.size()
	scaler = saved_scaler
	return true


# === The older names === #
# Every method above used to carry a leading underscore, which in Godot marks a
# method as virtual or private: the engine calls _ready() and _process(), you do not.
# The names below are the ones that shipped, kept working so nothing that already
# calls them breaks. They only forward. Prefer the ones without the underscore.

func _fit(newX, newY):
	return fit(newX, newY)

func _predict(newX):
	return predict(newX)



# === End SVM model === #
