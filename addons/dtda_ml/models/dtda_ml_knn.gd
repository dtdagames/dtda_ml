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
	X = newX
	Y = newY

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

# === End KNN model === #
