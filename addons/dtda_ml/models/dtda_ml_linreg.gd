extends DTDATools

class_name DTDALinReg

# === Linear Regression model === #
var m: int = 0
var n: int = 0
var rate: float
var iterations: int
var W
var b
var X
var Y
# features and target are standardized before the descent
var x_scaler
var y_scaler

func _init(newRate: float, newIterations: int) -> void:
	rate = newRate
	iterations = newIterations

# prediction in the standardized space, used by the descent
func _predict_scaled(scaledX, useW, useB) -> Array:
	var dotXW = _dot_product(scaledX, useW)
	return _add_arrays_const(dotXW, useB)

# one round of the descent, on the weights the work is holding rather than the ones
# the model is still answering with
func _update_weights(work: Dictionary) -> void:
	var rows = work["X"]
	var Y_pred: Array = _predict_scaled(rows, work["W"], work["b"])
	# gradients
	var subY: Array = _substract_arrays(work["Y"], Y_pred)
	var dotXY: Array = _dot_product(_transpose_array(rows), subY)
	var dotXY2: Array = _multiply_array_coef(dotXY, -2)
	var dW: Array = _divide_array_coef(dotXY2, work["m"])
	var sumY: float = _sum_array(subY)
	var dB: float = (-2.0 * sumY) / float(work["m"])
	# update
	var dWrate: Array = _multiply_array_coef(dW, rate)
	work["W"] = _substract_arrays(work["W"], dWrate)
	work["b"] = work["b"] - (rate * dB)

func fit_begin(newX, newY) -> bool:
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
	# the slice is one round of the descent, and the rounds run on their own copy:
	# a training abandoned halfway leaves the model with the weights it already had
	var pending_x := DTDAScaler.new()
	var pending_y := DTDAScaler.new()
	_fit_work = {
		"m": newX.size(),
		"n": newX[0].size(),
		"W": _array_zeros(newX[0].size()),
		"b": 0.0,
		"X": pending_x.fit_transform(newX),
		"Y": _matrix_to_column(pending_y.fit_transform(_column_to_matrix(newY))),
		"x_scaler": pending_x,
		"y_scaler": pending_y,
		"round": 0,
	}
	return true

func _model_name() -> String:
	return "DTDALinReg"

func fit_step() -> float:
	if _fit_work == null:
		push_error("DTDALinReg: fit_step() called with no training under way")
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
	x_scaler = work["x_scaler"]
	y_scaler = work["y_scaler"]
	_fit_work = null
	return 1.0

# training in one go: begin, then step until there is nothing left
func fit(newX, newY) -> bool:
	if not fit_begin(newX, newY):
		return false
	return _fit_every_step()

func predict(newX) -> Array:
	if not _check_fitted("DTDALinReg", W):
		return []
	var pred = _predict_scaled(x_scaler.transform(newX), W, b)
	# back to the unit of the training target
	return _matrix_to_column(y_scaler.inverse_transform(_column_to_matrix(pred)))

func to_dict():
	if not _check_fitted("DTDALinReg", W, "save()"):
		return {}
	return {
		"model": "DTDALinReg",
		"version": 1,
		"rate": rate,
		"iterations": iterations,
		"W": W,
		"b": b,
		"x_scaler": x_scaler.to_dict(),
		"y_scaler": y_scaler.to_dict(),
	}

func from_dict(data):
	if not _check_model_name(data, "DTDALinReg"):
		return false
	# Everything is read aside first and only takes the place of the standing model
	# once the whole file is known to be readable. Assigning as it went used to leave
	# a working model with the weights of a file it had just refused
	# absent, malformed and unusable answer the same way: a null is not a list either,
	# and predict() computes with every one of these numbers
	var saved_W = data.get("W")
	if not _check_number_array(saved_W, "DTDALinReg", "weights"):
		return false
	var saved_b = data.get("b", 0)
	if not _check_number(saved_b, "DTDALinReg", "intercept"):
		return false
	var saved_x = DTDAScaler.new()
	var saved_y = DTDAScaler.new()
	if not saved_x.from_dict(data.get("x_scaler", {})):
		return false
	if not saved_y.from_dict(data.get("y_scaler", {})):
		return false
	rate = data.get("rate", rate)
	iterations = data.get("iterations", iterations)
	W = saved_W
	b = saved_b
	n = W.size()
	x_scaler = saved_x
	y_scaler = saved_y
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



# === End Linear Regression model === #
