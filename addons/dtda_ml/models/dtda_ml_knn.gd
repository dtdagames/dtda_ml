extends MLTools

class_name DTDAKNN

# === KNN model === #
var X
var Y
var num_neighbors

func _init(n:int):
	num_neighbors=n

func _euclidean_distance(row1, row2):
	var distance = 0.0
	for i in row1.size():
		distance += (row1[i] - row2[i])**2
	return sqrt(distance)

func _get_neighbors(test_row):
	var distances = []
	var i = 0
	for train_row in X:
		var dist = _euclidean_distance(test_row, train_row)
		distances.push_back([i, dist])
		i += 1
	distances.sort_custom(func(a, b): return a[1] < b[1])
	var neighbors = []
	var tempNeighbors = num_neighbors
	if num_neighbors > X.size():
		tempNeighbors = X.size()
	for u in tempNeighbors:
		neighbors.push_back(distances[u][0])
	return neighbors

func _fit(newX, newY):
	# The rows are weighed before a single field is written: a fit that took them as
	# they came would leave a working model holding a nan, or half rewritten by a
	# raise in the middle. Answers false when it refuses, true when it fitted
	if not _check_matrix(newX, "DTDAKNN"):
		return false
	if not _check_labels(newX, newY, "DTDAKNN"):
		return false
	X = newX
	Y = newY
	return true

# most frequent label among the neighbors, ties go to the closest one
func _majority_vote(output_values):
	var counts = {}
	for value in output_values:
		counts[value] = counts.get(value, 0) + 1
	var tempPred = output_values[0]
	var best_count = 0
	# output_values is ordered from the closest to the farthest neighbor,
	# so a strict comparison keeps the closest label on equality
	for value in output_values:
		if counts[value] > best_count:
			tempPred = value
			best_count = counts[value]
	return tempPred

func _predict(newX):
	if not _check_fitted("DTDAKNN", X):
		return []
	var pred = []
	for i in newX.size():
		var neighbors = _get_neighbors(newX[i])
		var output_values = []
		for rowId in neighbors:
			output_values.push_back(Y[rowId])
		pred.push_back(_majority_vote(output_values))
	return pred

func _to_dict():
	if not _check_fitted("DTDAKNN", X, "_save()"):
		return {}
	return {
		"model": "DTDAKNN",
		"version": 1,
		"num_neighbors": num_neighbors,
		"X": X,
		"Y": Y,
	}

func _from_dict(data):
	if not _check_model_name(data, "DTDAKNN"):
		return false
	# Everything is read aside first and only takes the place of the standing model
	# once the whole file is known to be readable. A file lives in user://, where it
	# can be edited by hand, and a training set that is a text used to load with a
	# success and fall apart at the first prediction
	# _get_neighbors() counts this out at every prediction, so it belongs with X and Y
	# and not with the settings _fit() reads: a text answered null, and a count of
	# zero or less answered a list of nulls, in both cases after loading with a success
	var saved_k = data.get("num_neighbors", num_neighbors)
	if not _check_number(saved_k, "DTDAKNN", "neighbour count"):
		return false
	if saved_k < 1:
		push_error("DTDAKNN: the saved neighbour count is %s, it takes at least one" % saved_k)
		return false
	var saved_X = data.get("X")
	var saved_Y = data.get("Y")
	if typeof(saved_X) != TYPE_ARRAY or typeof(saved_Y) != TYPE_ARRAY:
		push_error("DTDAKNN: the saved model has no training set")
		return false
	if saved_X.size() == 0 or saved_X.size() != saved_Y.size():
		push_error("DTDAKNN: the saved model holds %d rows and %d labels" % [saved_X.size(), saved_Y.size()])
		return false
	# _euclidean_distance() subtracts one row from another, column by column. The
	# labels are left alone, a KNN answers them as they come and they can be anything
	for row in saved_X:
		if not _check_number_array(row, "DTDAKNN", "training row"):
			return false
	num_neighbors = saved_k
	X = saved_X
	Y = saved_Y
	return true

# === End KNN model === #
