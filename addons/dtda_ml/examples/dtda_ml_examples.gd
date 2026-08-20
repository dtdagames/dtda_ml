extends Node

var mltools

var dataKNN = [
	[2, 4, 2, 1, 0, 0, 3],
	[2, 2, 4, 0, 0, 0, 4],
	[4, 2, 1, 1, 0, 1, 5],
	[2, 2, 4, 0, 1, 1, 6],
]

var dataLinR = [
	[1.6, 40000],
	[4.6, 60000],
	[4.2, 58000],
	[4.1, 59000],
	[5.4, 80000],
	[8.1, 100000],
	[8.9, 110000],
	[9.2, 110000],
	[9.3, 114000],
	[10.2, 121000],
]

var dataLogR = [
	[2, 4, 2, 1, 0, 0, 0],
	[2, 2, 4, 0, 0, 0, 0],
	[4, 2, 1, 1, 0, 1, 1],
	[2, 2, 4, 0, 1, 1, 1],
]

var dataSVM = [
	[2, 4, 2, 1, 0, 0, 0],
	[2, 2, 4, 0, 0, 0, 0],
	[4, 2, 1, 1, 0, 1, 1],
	[2, 2, 4, 0, 1, 1, 1],
]

func _ready():
	mltools = DTDATools.new()
	
	_knn_example()
	_linreg_example()
	_logreg_example()
	_svm_example()
	_tree_example()
	_forest_example()
	_kmeans_example()
	_qlearning_example()
	_scaler_example()
	_metrics_example()
	_persistence_example()

func _knn_example():
	var X_train = mltools.drop_variable(dataKNN, dataKNN[0].size()-1)
	var y_train = mltools.get_variable(dataKNN, dataKNN[0].size()-1)
	var X_test = [
		[1, 4, 1, 1, 0, 0],
		[2, 2, 4, 1, 1, 1],
		[4, 1, 1, 0, 1, 0],
	]
	var y_test = [
		3,
		6,
		5,
	]
	
	var knn = DTDAKNN.new(3)
	knn.fit(X_train, y_train)
	print("KNN predictions: ", knn.predict(X_test))
	print("KNN score: ", mltools.get_perf(knn.predict(X_test), y_test, 0), "%")

func _linreg_example():
	var X_train = mltools.drop_variable(dataLinR, dataLinR[0].size()-1)
	var y_train = mltools.get_variable(dataLinR, dataLinR[0].size()-1)
	var X_test = [
		[7.2],
		[9.0],
		[11.1],
	]
	
	var linreg = DTDALinReg.new(0.01, 1000)
	linreg.fit(X_train, y_train)
	print("Linear Regression predictions: ", linreg.predict(X_test))

func _logreg_example():
	var X_train = mltools.drop_variable(dataLogR, dataLogR[0].size()-1)
	var Y_train = mltools.get_variable(dataLogR, dataLogR[0].size()-1)
	var X_test = [
		[1, 3, 1, 0, 1, 0],
		[2, 2, 4, 1, 1, 1],
		[4, 1, 1, 0, 1, 0],
	]
	var y_test = [
		0,
		1,
		1,
	]
	
	var logreg = DTDALogReg.new(0.01, 1000)
	logreg.fit(X_train, Y_train)
	print("Logistic Regression predictions: ", logreg.predict(X_test))
	print("Logistic Regression score: ", mltools.get_perf(logreg.predict(X_test), y_test, 2), "%")

func _svm_example():
	var X_train = mltools.drop_variable(dataSVM, dataSVM[0].size()-1)
	var Y_train = mltools.get_variable(dataSVM, dataSVM[0].size()-1)
	var X_test = [
		[1, 3, 1, 0, 1, 0],
		[2, 2, 4, 1, 1, 1],
		[4, 1, 1, 0, 1, 0],
	]
	var y_test = [
		0,
		1,
		1,
	]
	
	var svm = DTDASVM.new(0.01, 0.01, 1000)
	svm.fit(X_train, Y_train)
	print("SVM predictions: ", svm.predict(X_test))
	print("SVM score: ", mltools.get_perf(svm.predict(X_test), y_test, 3), "%")

func _tree_example():
	# classification, the tree answers the majority label of the leaf
	var X_train = mltools.drop_variable(dataLogR, dataLogR[0].size()-1)
	var Y_train = mltools.get_variable(dataLogR, dataLogR[0].size()-1)
	var X_test = [
		[1, 3, 1, 0, 1, 0],
		[2, 2, 4, 1, 1, 1],
		[4, 1, 1, 0, 1, 0],
	]
	var y_test = [0, 1, 1]

	var tree = DTDATree.new(3, 2, DTDATree.CLASSIFIER)
	tree.fit(X_train, Y_train)
	print("Tree predictions: ", tree.predict(X_test))
	print("Tree score: ", mltools.accuracy(tree.predict(X_test), y_test), "%")

	# regression, the tree answers the mean of the leaf
	var X_lin = mltools.drop_variable(dataLinR, dataLinR[0].size()-1)
	var y_lin = mltools.get_variable(dataLinR, dataLinR[0].size()-1)
	var regressor = DTDATree.new(3, 2, DTDATree.REGRESSOR)
	regressor.fit(X_lin, y_lin)
	print("Tree regression: ", regressor.predict([[7.2], [9.0], [11.1]]))
	print("Tree R2: ", mltools.r2_score(regressor.predict(X_lin), y_lin))

	# a XOR, which no linear model can separate
	var xor_X = [[0, 0], [0, 1], [1, 0], [1, 1]]
	var xor_y = [0, 1, 1, 0]
	var xor_tree = DTDATree.new(4, 2, DTDATree.CLASSIFIER)
	xor_tree.fit(xor_X, xor_y)
	print("Tree on a XOR: ", xor_tree.predict(xor_X), " expected ", xor_y)

# A crowd of trees, each grown on its own draw of the rows and of the features.
# The world below has one feature that decides the label and five that are pure noise,
# and one row in eight carries the wrong label. That is what a lone deep tree learns
# by heart, and what a forest refuses to.
func _forest_rows(first, count, flip_every):
	var X = []
	var y = []
	for k in count:
		var i = first + k
		var x0 = i % 20
		var row = [x0]
		for j in 5:
			row.push_back((i * 37 + (j + 1) * 53) % 13)
		var label = 1 if x0 >= 10 else 0
		if flip_every > 0 and i % flip_every == 3:
			label = 1 - label
		X.push_back(row)
		y.push_back(label)
	return [X, y]

func _forest_example():
	var train = _forest_rows(0, 48, 8)
	var unseen = _forest_rows(1000, 48, 0)

	var tree = DTDATree.new(8, 2, DTDATree.CLASSIFIER)
	tree.fit(train[0], train[1])
	print("Lone tree, on the rows it learned: ", mltools.accuracy(tree.predict(train[0]), train[1]), "%")
	print("Lone tree, on rows it never saw: ", mltools.accuracy(tree.predict(unseen[0]), unseen[1]), "%")

	# a test set of 48 rows means one row is worth two points, so a single forest can
	# land level with the tree by luck. Five of them, seeded so the run repeats, is
	# what the picture actually looks like
	var total = 0.0
	for k in 5:
		var forest = DTDAForest.new(25, 8, 2, DTDAForest.CLASSIFIER)
		forest.set_seed(k + 1)
		forest.fit(train[0], train[1])
		total += mltools.accuracy(forest.predict(unseen[0]), unseen[1])
	print("Forest, on rows it never saw, five seeds averaged: ", snapped(total / 5.0, 0.01), "%")

	# regression, where the trees are averaged instead of voting
	var X_lin = mltools.drop_variable(dataLinR, dataLinR[0].size()-1)
	var y_lin = mltools.get_variable(dataLinR, dataLinR[0].size()-1)
	var regressor = DTDAForest.new(15, 4, 2, DTDAForest.REGRESSOR)
	regressor.set_seed(1)
	regressor.fit(X_lin, y_lin)
	print("Forest regression: ", regressor.predict([[7.2], [9.0], [11.1]]))
	print("Forest R2: ", mltools.r2_score(regressor.predict(X_lin), y_lin))

# The one model here that is given no labels at all: it is handed positions on a map
# and works out which camp each one belongs to. Three camps are planted below and
# nothing says so, K-Means has to find them.
func _kmeans_rows():
	var camps = [[0.0, 0.0], [40.0, 5.0], [20.0, 35.0]]
	var X = []
	for c in camps.size():
		for j in 8:
			var i = c * 8 + j
			X.push_back([camps[c][0] + ((i * 7) % 5) * 1.5 - 3.0,
				camps[c][1] + ((i * 11) % 5) * 1.5 - 3.0])
	return X

func _kmeans_example():
	var X = _kmeans_rows()
	var kmeans = DTDAKMeans.new(3)
	# a seed, so this example prints the same fit every time
	kmeans.set_seed(1)
	print("K-Means groups: ", kmeans.fit_predict(X))
	var camps = kmeans.get_centroids()
	for c in camps.size():
		print("K-Means camp ", c, " at: ", snapped(camps[c][0], 0.01), ", ", snapped(camps[c][1], 0.01))
	print("K-Means inertia: ", snapped(kmeans.inertia, 0.001))

	# how tight the grouping is for each k. Inertia always falls as k rises, so it is
	# read for where it stops falling sharply, not for how low it goes
	for count in range(1, 6):
		var trial = DTDAKMeans.new(count)
		trial.set_seed(1)
		trial.fit(X)
		print("K-Means with ", count, " groups, inertia ", snapped(trial.inertia, 0.001))

	# a brand new spot, put with the camp it is nearest to
	print("K-Means places a new point: ", kmeans.predict([[38.0, 6.0]]))

# a corridor of six rooms: room 0 is a pit, room 5 is the exit, the agent starts in room 3
# it learns by playing, there is no training set here
const CORRIDOR_ACTIONS = ["left", "right"]
const CORRIDOR_PIT = 0
const CORRIDOR_EXIT = 5
const CORRIDOR_START = 3

# returns [next room, reward, episode over]
func _corridor_step(room, action):
	var next_room = clamp(room + (1 if action == "right" else -1), CORRIDOR_PIT, CORRIDOR_EXIT)
	if next_room == CORRIDOR_EXIT:
		return [next_room, 1.0, true]
	if next_room == CORRIDOR_PIT:
		return [next_room, -1.0, true]
	return [next_room, 0.0, false]

func _qlearning_example():
	var agent = DTDAQLearning.new(0.2, 0.9, 1.0, 0.99, 0.05)
	# a fixed seed so this example prints the same run every time
	agent.set_seed(1)

	for episode in 500:
		var room = CORRIDOR_START
		# a bounded episode, a random walk could otherwise wander for a long time
		for step in 100:
			var action = agent.choose_action(room, CORRIDOR_ACTIONS)
			var result = _corridor_step(room, action)
			agent.learn(room, action, result[1], result[0], CORRIDOR_ACTIONS, result[2])
			room = result[0]
			if result[2]:
				break
		# one notch less exploration, never below the floor
		agent.decay_exploration()

	print("Q-Learning exploration left: ", agent.exploration_rate)
	# the policy, with no exploration at all: every room walks away from the pit
	for room in range(1, CORRIDOR_EXIT):
		print("Q-Learning room ", room, ": ", agent.predict(room),
			" (left ", snapped(agent.get_q(room, "left"), 0.001),
			", right ", snapped(agent.get_q(room, "right"), 0.001), ")")

	# playing the learned policy, which should reach the exit in two moves
	var path = [CORRIDOR_START]
	var current = CORRIDOR_START
	for step in 10:
		var move = agent.predict(current, CORRIDOR_ACTIONS)
		# predict() answers null on a room the agent never visited, where it has
		# nothing to say: in a real game that is where you fall back on your own default
		if move == null:
			break
		var result = _corridor_step(current, move)
		current = result[0]
		path.push_back(current)
		if result[2]:
			break
	print("Q-Learning path from room ", CORRIDOR_START, ": ", path)

# the models scale their features on their own, use DTDAScaler for your own data
func _scaler_example():
	var raw = [
		[1.6, 40000],
		[5.4, 80000],
		[10.2, 121000],
	]

	var standard = DTDAScaler.new()
	print("Standardized: ", standard.fit_transform(raw))

	var minmax = DTDAScaler.new(DTDAScaler.MINMAX)
	var scaled = minmax.fit_transform(raw)
	print("Min-max: ", scaled)
	print("Back to the original unit: ", minmax.inverse_transform(scaled))

func _metrics_example():
	# classification, on the logistic regression data
	var X_train = mltools.drop_variable(dataLogR, dataLogR[0].size()-1)
	var Y_train = mltools.get_variable(dataLogR, dataLogR[0].size()-1)
	var X_test = [
		[1, 3, 1, 0, 1, 0],
		[2, 2, 4, 1, 1, 1],
		[4, 1, 1, 0, 1, 0],
	]
	var y_test = [0, 1, 1]

	var logreg = DTDALogReg.new(0.01, 1000)
	logreg.fit(X_train, Y_train)
	var y_pred = logreg.predict(X_test)
	print("Accuracy: ", mltools.accuracy(y_pred, y_test), "%")
	print("Confusion matrix: ", mltools.confusion_matrix(y_pred, y_test))
	print("Precision: ", mltools.precision(y_pred, y_test))
	print("Recall: ", mltools.recall(y_pred, y_test))
	print("F1 score: ", mltools.f1_score(y_pred, y_test))

	# regression, scored on the training set itself
	var X_lin = mltools.drop_variable(dataLinR, dataLinR[0].size()-1)
	var y_lin = mltools.get_variable(dataLinR, dataLinR[0].size()-1)
	var linreg = DTDALinReg.new(0.01, 1000)
	linreg.fit(X_lin, y_lin)
	var lin_pred = linreg.predict(X_lin)
	print("R2: ", mltools.r2_score(lin_pred, y_lin))
	print("RMSE: ", mltools.rmse(lin_pred, y_lin))
	print("MAE: ", mltools.mae(lin_pred, y_lin))

# train once, ship the weights, predict without the training set
func _persistence_example():
	var path = "user://dtda_linreg.json"
	var X_train = mltools.drop_variable(dataLinR, dataLinR[0].size()-1)
	var y_train = mltools.get_variable(dataLinR, dataLinR[0].size()-1)
	var X_test = [
		[7.2],
		[9.0],
		[11.1],
	]

	var linreg = DTDALinReg.new(0.01, 1000)
	linreg.fit(X_train, y_train)
	print("Before saving: ", linreg.predict(X_test))
	if not linreg.save(path):
		return

	# a brand new model, never fitted
	var loaded = DTDALinReg.new(0.01, 1000)
	if loaded.load(path):
		print("After loading: ", loaded.predict(X_test))
