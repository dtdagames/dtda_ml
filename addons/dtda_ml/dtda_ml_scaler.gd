extends DTDATools

class_name DTDAScaler

# === Feature scaler === #
# STANDARD centers each column on its mean and divides it by its standard deviation
# MINMAX brings each column into the [0, 1] range
enum { STANDARD, MINMAX }

var mode
# each column is scaled as (value - offset) / scale
var offsets
var scales

func _init(scaler_mode := STANDARD):
	mode = scaler_mode

# learn the offset and the scale of every column
func fit(X):
	if X.size() == 0:
		push_error("DTDAScaler: fit() called with no data")
		return
	offsets = []
	scales = []
	for column in _transpose_array(X):
		if mode == MINMAX:
			var low = column.min()
			var high = column.max()
			# float() is load bearing: on a column of integers, high - low would stay
			# an integer and _transform would then do an integer division, so a value
			# such as 40000 / 81000 would come out as 0 instead of 0.49
			offsets.push_back(float(low))
			# a constant column would divide by zero
			scales.push_back(1.0 if high == low else float(high - low))
		else:
			offsets.push_back(_mean_array(column))
			# _std_array already returns 1.0 on a constant column
			scales.push_back(_std_array(column))

func transform(X):
	if not _check_fitted("DTDAScaler", offsets, "transform()"):
		return []
	var matrix = []
	for i in X.size():
		matrix.push_back([])
		for u in X[i].size():
			matrix[i].push_back((X[i][u] - offsets[u]) / scales[u])
	return matrix

func fit_transform(X):
	fit(X)
	return transform(X)

# back to the unit of the data the scaler was fitted on
func inverse_transform(X):
	if not _check_fitted("DTDAScaler", offsets, "inverse_transform()"):
		return []
	var matrix = []
	for i in X.size():
		matrix.push_back([])
		for u in X[i].size():
			matrix[i].push_back(X[i][u] * scales[u] + offsets[u])
	return matrix

func to_dict():
	return {
		"mode": mode,
		"offsets": offsets,
		"scales": scales,
	}

# A saved scaler has to be usable, not merely present. transform() reads an offset
# and a scale per column and divides by the scale, so a file holding a string, a list
# shorter than the other or a zero used to load without a word and only fall apart at
# the first prediction, or worse answer inf. A model file lives in user://, where it
# can be edited by hand.
# Nothing is written into the scaler until the whole dictionary has been read, so a
# refused one leaves a working scaler exactly as it was
func from_dict(data):
	var saved_offsets = data.get("offsets")
	var saved_scales = data.get("scales")
	if typeof(saved_offsets) != TYPE_ARRAY or typeof(saved_scales) != TYPE_ARRAY:
		push_error("DTDAScaler: the saved scaler is incomplete")
		return false
	if saved_offsets.size() == 0 or saved_offsets.size() != saved_scales.size():
		push_error("DTDAScaler: the saved scaler holds %d offsets and %d scales" % [saved_offsets.size(), saved_scales.size()])
		return false
	for i in saved_offsets.size():
		if not (typeof(saved_offsets[i]) in [TYPE_INT, TYPE_FLOAT] and typeof(saved_scales[i]) in [TYPE_INT, TYPE_FLOAT]):
			push_error("DTDAScaler: the saved scaler holds something that is not a number")
			return false
		# transform() divides by this, and fit() never writes a zero there: a
		# constant column is given a scale of 1.0 for that very reason
		if float(saved_scales[i]) == 0.0:
			push_error("DTDAScaler: the saved scaler holds a scale of zero")
			return false
	# int() because a mode read back from JSON carries as a float
	mode = int(data.get("mode", STANDARD))
	offsets = saved_offsets
	scales = saved_scales
	return true


# === The older names === #
# Every method above used to carry a leading underscore, which in Godot marks a
# method as virtual or private: the engine calls _ready() and _process(), you do not.
# The names below are the ones that shipped, kept working so nothing that already
# calls them breaks. They only forward. Prefer the ones without the underscore.

func _fit(X):
	return fit(X)

func _transform(X):
	return transform(X)

func _fit_transform(X):
	return fit_transform(X)

func _inverse_transform(X):
	return inverse_transform(X)



# === End Feature scaler === #
