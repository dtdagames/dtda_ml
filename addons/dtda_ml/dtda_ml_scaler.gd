extends DTDATools

class_name DTDAScaler

# STANDARD centers each column on its mean and divides by its standard deviation, MINMAX brings it into [0, 1]
enum { STANDARD, MINMAX }

var mode: int
# each column is scaled as (value - offset) / scale
var offsets
var scales

func _init(scaler_mode: int = STANDARD) -> void:
	mode = scaler_mode

func fit(X) -> bool:
	if X.size() == 0:
		push_error("DTDAScaler: fit() called with no data")
		return false
	offsets = []
	scales = []
	for column in _transpose_array(X):
		if mode == MINMAX:
			var low = column.min()
			var high = column.max()
			# float() is load bearing: on a column of integers, high - low would stay an integer and _transform would then do an integer division, 40000 / 81000 coming out as 0 instead of 0.49. The 1.0 below is there because a constant column would divide by zero
			offsets.push_back(float(low))
			scales.push_back(1.0 if high == low else float(high - low))
		else:
			offsets.push_back(_mean_array(column))
			# _std_array already returns 1.0 on a constant column
			scales.push_back(_std_array(column))
	return true

func transform(X) -> Array:
	if not _check_fitted("DTDAScaler", offsets, "transform()"):
		return []
	var matrix: Array = []
	for i in X.size():
		matrix.push_back([])
		for u in X[i].size():
			matrix[i].push_back((X[i][u] - offsets[u]) / scales[u])
	return matrix

func fit_transform(X) -> Array:
	fit(X)
	return transform(X)

func inverse_transform(X) -> Array:
	if not _check_fitted("DTDAScaler", offsets, "inverse_transform()"):
		return []
	var matrix: Array = []
	for i in X.size():
		matrix.push_back([])
		for u in X[i].size():
			matrix[i].push_back(X[i][u] * scales[u] + offsets[u])
	return matrix

func to_dict() -> Dictionary:
	return {
		"mode": mode,
		"offsets": offsets,
		"scales": scales,
	}

# A saved scaler has to be usable, not merely present: a file holding a string, a list shorter than the other or a zero scale used to load without a word and fall apart at the first prediction, or answer inf. Nothing is written into the scaler until the whole dictionary has been read, so a refused one leaves a working scaler exactly as it was.
func from_dict(data) -> bool:
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
		# transform() divides by this, and fit() never writes a zero there: a constant column is given a scale of 1.0 for that very reason
		if float(saved_scales[i]) == 0.0:
			push_error("DTDAScaler: the saved scaler holds a scale of zero")
			return false
	# int() because a mode read back from JSON carries as a float
	mode = int(data.get("mode", STANDARD))
	offsets = saved_offsets
	scales = saved_scales
	return true


# the older underscored spellings, kept working for what already calls them; they only forward

func _fit(X):
	return fit(X)

func _transform(X):
	return transform(X)

func _fit_transform(X):
	return fit_transform(X)

func _inverse_transform(X):
	return inverse_transform(X)

