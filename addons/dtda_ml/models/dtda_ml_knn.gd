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
	for i in row1.size()-1:
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
	for u in num_neighbors:
		neighbors.push_back(distances[u][0])
	return neighbors

func _fit(newX, newY):
	X = newX
	Y = newY

func _predict(newX):
	var pred = []
	for i in newX.size():
		var neighbors = _get_neighbors(newX[i])
		var output_values = []
		for rowId in neighbors:
			output_values.push_back(Y[rowId])
		var tempPred = output_values[0]
		pred.push_back(tempPred)
	return pred

# === End KNN model === #
