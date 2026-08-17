extends MLTools

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
func _fit(X):
	if X.size() == 0:
		push_error("DTDAScaler: _fit() called with no data")
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

func _transform(X):
	if not _check_fitted("DTDAScaler", offsets, "_transform()"):
		return []
	var matrix = []
	for i in X.size():
		matrix.push_back([])
		for u in X[i].size():
			matrix[i].push_back((X[i][u] - offsets[u]) / scales[u])
	return matrix

func _fit_transform(X):
	_fit(X)
	return _transform(X)

# back to the unit of the data the scaler was fitted on
func _inverse_transform(X):
	if not _check_fitted("DTDAScaler", offsets, "_inverse_transform()"):
		return []
	var matrix = []
	for i in X.size():
		matrix.push_back([])
		for u in X[i].size():
			matrix[i].push_back(X[i][u] * scales[u] + offsets[u])
	return matrix

func _to_dict():
	return {
		"mode": mode,
		"offsets": offsets,
		"scales": scales,
	}

func _from_dict(data):
	mode = data.get("mode", STANDARD)
	offsets = data.get("offsets")
	scales = data.get("scales")
	if offsets == null or scales == null:
		push_error("DTDAScaler: the saved scaler is incomplete")
		return false
	return true

# === End Feature scaler === #
