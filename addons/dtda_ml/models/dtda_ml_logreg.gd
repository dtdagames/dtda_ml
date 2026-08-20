extends DTDATools

class_name DTDALogReg

# === Logistic Regression model === #
var m: int = 0
var n: int = 0
var rate: float
var iterations: int
var W
var b
var X
var Y
var scaler

func _init(newRate: float, newIterations: int) -> void:
	rate = newRate
	iterations = newIterations

# one round of the descent, on the weights the work is holding rather than the ones
# the model is still answering with
func _update_weights(work: Dictionary) -> void:
	var rows = work["X"]
	var a = _sigmoid(rows, work["W"], work["b"])

	# gradients
	var tmp = _substract_arrays(a, _transpose_simple_array(work["Y"]))
	var dotW = _dot_product(_transpose_array(rows), tmp)
	var dW = _divide_array_coef(dotW, work["m"])
	var dB = _sum_array(tmp) / float(work["m"])

	# update
	var dWrate = _multiply_array_coef(dW, rate)
	work["W"] = _substract_arrays(work["W"], dWrate)
	work["b"] = work["b"] - (rate * dB)

func fit_begin(newX, newY) -> bool:
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
	# the slice is one round of the descent, and the rounds run on their own copy:
	# a training abandoned halfway leaves the model with the weights it already had
	# standardized features, otherwise exp() overflows as soon as the data gets large
	var pending := DTDAScaler.new()
	_fit_work = {
		"m": newX.size(),
		"n": newX[0].size(),
		"W": _array_zeros(newX[0].size()),
		"b": 0.0,
		"X": pending.fit_transform(newX),
		"Y": newY,
		"scaler": pending,
		"round": 0,
	}
	return true

func _model_name() -> String:
	return "DTDALogReg"

func fit_step() -> float:
	if _fit_work == null:
		push_error("DTDALogReg: fit_step() called with no training under way")
		return 1.0
	var work: Dictionary = _fit_work
	if int(work["round"]) < iterations:
		_update_weights(work)
		work["round"] = int(work["round"]) + 1
	if int(work["round"]) < iterations:
		return float(work["round"]) / float(iterations)
	m = work["m"]
	n = work["n"]
	W = work["W"]
	b = work["b"]
	X = work["X"]
	Y = work["Y"]
	scaler = work["scaler"]
	_fit_work = null
	return 1.0

# training in one go: begin, then step until there is nothing left
func fit(newX, newY) -> bool:
	if not fit_begin(newX, newY):
		return false
	return _fit_every_step()

func _sigmoid(newX, newW, newB) -> Array:
	# 1/(1 + e(-(x.dot(w) + b)
	var dotXW = _dot_product(newX, newW)
	var dotXWb = _add_arrays_const(dotXW, newB)
	var dotXWbn = _multiply_array_coef(dotXWb, -1)
	var expXWb = _exp_array_(dotXWbn)
	var expXWb1 = _add_arrays_const(expXWb, 1)
	return _divide_inverse_array_coef(expXWb1, 1)

func predict(newX) -> Array:
	if not _check_fitted("DTDALogReg", W):
		return []
	var Z = _sigmoid(scaler.transform(newX), W, b)
	var matrix = []
	for i in Z:
		matrix.push_back(round(i))
	return matrix

func to_dict():
	if not _check_fitted("DTDALogReg", W, "save()"):
		return {}
	return {
		"model": "DTDALogReg",
		"version": 1,
		"rate": rate,
		"iterations": iterations,
		"W": W,
		"b": b,
		"scaler": scaler.to_dict(),
	}

func from_dict(data):
	if not _check_model_name(data, "DTDALogReg"):
		return false
	# Everything is read aside first and only takes the place of the standing model
	# once the whole file is known to be readable. Assigning as it went used to leave
	# a working model with the weights of a file it had just refused
	# absent, malformed and unusable answer the same way: a null is not a list either,
	# and predict() computes with every one of these numbers
	var saved_W = data.get("W")
	if not _check_number_array(saved_W, "DTDALogReg", "weights"):
		return false
	var saved_b = data.get("b", 0)
	if not _check_number(saved_b, "DTDALogReg", "intercept"):
		return false
	var saved_scaler = DTDAScaler.new()
	if not saved_scaler.from_dict(data.get("scaler", {})):
		return false
	rate = data.get("rate", rate)
	iterations = data.get("iterations", iterations)
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



# === End Logistic Regression model === #
