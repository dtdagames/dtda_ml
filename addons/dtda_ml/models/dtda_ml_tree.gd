extends MLTools

class_name DTDATree

# === Decision tree (CART) === #
# CLASSIFIER splits on the Gini impurity and a leaf answers the majority label
# REGRESSOR splits on the variance and a leaf answers the mean
enum { CLASSIFIER, REGRESSOR }

# max_features is what DTDAForest needs from a tree: how many features a single split
# may look at, drawn again at every node. 0, the default, means all of them, in order,
# and draws nothing at all, so a tree built on its own behaves exactly as it always did.
# A forest whose trees each looked at every feature would grow the same tree over and
# over, and averaging identical trees gains nothing.

var m
var n
var mode
var max_depth
var min_samples_split
var max_features
# its own generator, so a forest can hand each of its trees a reproducible stream
var rng
var X
var Y
# the tree itself, nested dictionaries of nodes
# a branch holds feature/threshold/left/right, a leaf holds a single value
var root

func _init(tree_max_depth := 5, tree_min_samples_split := 2, tree_mode := CLASSIFIER, tree_max_features := 0):
	max_depth = tree_max_depth
	min_samples_split = tree_min_samples_split
	mode = tree_mode
	max_features = tree_max_features
	rng = RandomNumberGenerator.new()

# fix the feature draws, for a reproducible tree. Pointless while max_features is 0,
# where nothing is drawn
func _set_seed(value):
	rng.seed = value

# Gini impurity of the labels held by the given rows, 0.0 when they all agree
func _gini(rows):
	var counts = {}
	for i in rows:
		counts[Y[i]] = counts.get(Y[i], 0) + 1
	var impurity = 1.0
	for label in counts:
		var p = float(counts[label]) / float(rows.size())
		impurity -= p * p
	return impurity

# variance of the labels held by the given rows
func _variance(rows):
	var values = []
	for i in rows:
		values.push_back(Y[i])
	var mean = _mean_array(values)
	var total = 0.0
	for value in values:
		total += (value - mean)**2
	return total / float(values.size())

func _impurity(rows):
	if rows.size() == 0:
		return 0.0
	if mode == REGRESSOR:
		return _variance(rows)
	return _gini(rows)

# every midpoint between two consecutive distinct values of a feature
func _candidate_thresholds(rows, feature):
	var values = []
	for i in rows:
		values.push_back(X[i][feature])
	values.sort()
	var thresholds = []
	for i in range(1, values.size()):
		if values[i] != values[i-1]:
			thresholds.push_back((values[i] + values[i-1]) / 2.0)
	return thresholds

# the features a single split may look at, all of them in order by default
func _features_for_split():
	var every = []
	for feature in n:
		every.push_back(feature)
	# the default path draws nothing, so it stays identical run after run
	if max_features <= 0 or max_features >= n:
		return every
	# without replacement, so no feature is weighed twice in the same split
	var drawn = []
	for i in max_features:
		drawn.push_back(every.pop_at(rng.randi() % every.size()))
	return drawn

# the split lowering the impurity the most, or an empty dictionary when none does
# when max_features hides every usable feature from a node, that node finds nothing
# and becomes a leaf. A lone tree keeps growing, only a forest can end up there
func _best_split(rows):
	var parent = _impurity(rows)
	var best = {}
	# a split of gain 0 is still worth taking: on a XOR, no single feature helps at the
	# root, yet each half becomes separable one level down. Only the absence of any
	# usable threshold leaves this empty.
	var best_gain = -1.0
	for feature in _features_for_split():
		for threshold in _candidate_thresholds(rows, feature):
			var left = []
			var right = []
			for i in rows:
				if X[i][feature] <= threshold:
					left.push_back(i)
				else:
					right.push_back(i)
			# a threshold taken between two distinct values always fills both sides,
			# the guard only protects against a malformed row
			if left.size() == 0 or right.size() == 0:
				continue
			var weighted = (left.size() * _impurity(left) + right.size() * _impurity(right)) / float(rows.size())
			var gain = parent - weighted
			if gain > best_gain:
				best_gain = gain
				best = {"feature": feature, "threshold": threshold, "left": left, "right": right}
	return best

# what a leaf answers: the mean in regression, the majority label otherwise
func _leaf_value(rows):
	var values = []
	for i in rows:
		values.push_back(Y[i])
	if mode == REGRESSOR:
		return _mean_array(values)
	var counts = {}
	for value in values:
		counts[value] = counts.get(value, 0) + 1
	var leaf = values[0]
	var best_count = 0
	for value in values:
		if counts[value] > best_count:
			leaf = value
			best_count = counts[value]
	return leaf

func _build(rows, depth):
	# a pure node, or one too small or too deep to be split again
	if depth >= max_depth or rows.size() < min_samples_split or _impurity(rows) == 0.0:
		return {"leaf": _leaf_value(rows)}
	var split = _best_split(rows)
	# no split lowers the impurity, for instance when identical rows carry different labels
	if split.is_empty():
		return {"leaf": _leaf_value(rows)}
	return {
		"feature": split["feature"],
		"threshold": split["threshold"],
		"left": _build(split["left"], depth + 1),
		"right": _build(split["right"], depth + 1),
	}

func _fit(newX, newY):
	if newX.size() == 0:
		push_error("DTDATree: _fit() called with no data")
		return
	m = newX.size()
	n = newX[0].size()
	X = newX
	Y = newY

	var rows = []
	for i in m:
		rows.push_back(i)
	root = _build(rows, 0)

# walk down the tree, a value lower than or equal to the threshold goes left
func _predict_row(row):
	var node = root
	while not node.has("leaf"):
		# int() because a tree read back from JSON carries its feature index as a float
		if row[int(node["feature"])] <= node["threshold"]:
			node = node["left"]
		else:
			node = node["right"]
	return node["leaf"]

func _predict(newX):
	if not _check_fitted("DTDATree", root):
		return []
	var pred = []
	for i in newX.size():
		pred.push_back(_predict_row(newX[i]))
	return pred

func _to_dict():
	if not _check_fitted("DTDATree", root, "_save()"):
		return {}
	return {
		"model": "DTDATree",
		"version": 1,
		"mode": mode,
		"max_depth": max_depth,
		"min_samples_split": min_samples_split,
		"max_features": max_features,
		"root": root,
	}

func _from_dict(data):
	if not _check_model_name(data, "DTDATree"):
		return false
	mode = int(data.get("mode", CLASSIFIER))
	max_depth = int(data.get("max_depth", max_depth))
	min_samples_split = int(data.get("min_samples_split", min_samples_split))
	# absent from the files written before the forest existed, and 0 is what they did
	max_features = int(data.get("max_features", 0))
	var saved_root = data.get("root")
	if saved_root == null:
		push_error("DTDATree: the saved model has no tree")
		return false
	# a model file lives in user://, where it can be edited by hand, and DTDAForest
	# hands whole subtrees straight to this function
	if typeof(saved_root) != TYPE_DICTIONARY:
		push_error("DTDATree: the saved tree is not a node")
		return false
	root = saved_root
	return true

# === End Decision tree === #
