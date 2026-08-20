extends DTDATools

class_name DTDAForest

# === Random forest === #
# A crowd of DTDATree, each grown on its own draw of the training set, answering
# together: the majority label in CLASSIFIER, the mean in REGRESSOR.
#
# Two draws make the trees disagree, and disagreeing is the whole point, since
# averaging identical trees gains nothing:
#  - bagging, each tree is fitted on as many rows as the training set holds, drawn
#    with replacement, so it sees about two thirds of it and a different two thirds
#  - a feature draw at every split, carried by DTDATree.max_features
# Both come from one generator, so set_seed() replays a whole forest.
#
# Like a lone tree, a forest compares features to thresholds and needs no scaling.

# the modes of DTDATree, taken from it rather than spelled again, so the two
# cannot drift apart
const CLASSIFIER = DTDATree.CLASSIFIER
const REGRESSOR = DTDATree.REGRESSOR

const FORMAT_VERSION = 1

var num_trees: int
var max_depth: int
var min_samples_split: int
var mode: int
var max_features: int
var m: int = 0
var n: int = 0
# the trees themselves, null until fit()
var trees
# its own generator, so a run can be replayed with set_seed()
var rng: RandomNumberGenerator
# the seed given to set_seed(), replayed by reset(), null when none was asked for
var start_seed

func _init(forest_num_trees: int = 10, forest_max_depth: int = 5, forest_min_samples_split: int = 2, forest_mode: int = CLASSIFIER, forest_max_features: int = 0) -> void:
	num_trees = forest_num_trees
	max_depth = forest_max_depth
	min_samples_split = forest_min_samples_split
	mode = forest_mode
	max_features = forest_max_features
	rng = RandomNumberGenerator.new()
	start_seed = null

# fix the draws, for a reproducible forest
# reset() puts the generator back on that same seed
func set_seed(value: int) -> void:
	start_seed = value
	rng.seed = value

# forget the trees and put the generator back where it started
func reset() -> void:
	trees = null
	if start_seed != null:
		rng.seed = start_seed

# how many features one split may look at. 0, the default, asks for the usual rule:
# the square root of the count when classifying, a third of it when regressing.
# Passing the full count turns the forest into plain bagging, which is a fair thing
# to want and a poor default, the trees then being nearly the same tree
func _resolved_max_features(count: int) -> int:
	if max_features > 0:
		return min(max_features, count)
	if mode == REGRESSOR:
		# a real division, floored: a third of five features is one, never none
		return max(1, int(count / 3.0))
	return max(1, int(sqrt(float(count))))

func fit(newX, newY) -> bool:
	# The rows are weighed before a single field is written: a fit that took them as
	# they came would leave a working model holding a nan, or half rewritten by a
	# raise in the middle. Answers false when it refuses, true when it fitted
	if not _check_matrix(newX, "DTDAForest"):
		return false
	if not _check_labels(newX, newY, "DTDAForest"):
		return false
	# as in a lone tree: a leaf answers the mean when regressing, and only counts
	# labels when classifying, where a label can be whatever names a class
	if mode == REGRESSOR and not _check_number_array(newY, "DTDAForest", "labels"):
		return false
	if num_trees <= 0:
		push_error("DTDAForest: fit() called for %d trees" % num_trees)
		return false
	m = newX.size()
	n = newX[0].size()
	var per_split = _resolved_max_features(n)
	var grown: Array = []
	for i in num_trees:
		var bag_X: Array = []
		var bag_Y: Array = []
		# as many rows as the set holds, drawn with replacement: some rows land in
		# the bag twice, others not at all, which is what makes this tree its own
		for u in m:
			var pick = rng.randi() % m
			bag_X.push_back(newX[pick])
			bag_Y.push_back(newY[pick])
		var tree = DTDATree.new(max_depth, min_samples_split, mode, per_split)
		# each tree draws its features from a stream of ours, so the whole forest
		# replays from a single set_seed()
		tree.set_seed(rng.randi())
		tree.fit(bag_X, bag_Y)
		grown.push_back(tree)
	trees = grown
	return true

# how the trees are put back together: the mean when regressing, the majority label
# otherwise. A tie goes to the label the first tree answered, so the same forest
# answers the same thing every time it is asked
func _combine(answers):
	if mode == REGRESSOR:
		return _mean_array(answers)
	var counts = {}
	for value in answers:
		counts[value] = counts.get(value, 0) + 1
	var winner = answers[0]
	var best_count = 0
	for value in answers:
		if counts[value] > best_count:
			winner = value
			best_count = counts[value]
	return winner

func predict(newX) -> Array:
	if not _check_fitted("DTDAForest", trees):
		return []
	# every tree answers the whole batch, then each row is settled across the trees
	var answers: Array = []
	for tree in trees:
		answers.push_back(tree.predict(newX))
	var pred: Array = []
	for i in newX.size():
		var row = []
		for tree_answers in answers:
			row.push_back(tree_answers[i])
		pred.push_back(_combine(row))
	return pred

func to_dict() -> Dictionary:
	if not _check_fitted("DTDAForest", trees, "save()"):
		return {}
	var saved = []
	for tree in trees:
		# a tree already knows how to write itself down, a forest is the list
		saved.push_back(tree.to_dict())
	return {
		"model": "DTDAForest",
		"version": FORMAT_VERSION,
		"mode": mode,
		"num_trees": num_trees,
		"max_depth": max_depth,
		"min_samples_split": min_samples_split,
		"max_features": max_features,
		"trees": saved,
	}

func from_dict(data) -> bool:
	if not _check_model_name(data, "DTDAForest"):
		return false
	# int() because a version read back from JSON carries as a float
	var version = int(data.get("version", 0))
	if version != FORMAT_VERSION:
		push_error("DTDAForest: this file is written in format %d, this model reads format %d" % [version, FORMAT_VERSION])
		return false
	var saved = data.get("trees")
	# absent and malformed answer the same way: a null is not a list either
	if typeof(saved) != TYPE_ARRAY:
		push_error("DTDAForest: the saved model has no list of trees")
		return false
	if saved.size() == 0:
		push_error("DTDAForest: the saved model holds no tree")
		return false
	# a model file lives in user://, where a player can edit it: the trees are rebuilt
	# one by one and only replace the standing ones once every last one is readable
	var rebuilt = []
	for entry in saved:
		if typeof(entry) != TYPE_DICTIONARY:
			push_error("DTDAForest: one of the saved trees is not a tree")
			return false
		var tree = DTDATree.new()
		# the tree checks its own name, its own root and its own structure
		if not tree.from_dict(entry):
			push_error("DTDAForest: one of the saved trees could not be read")
			return false
		rebuilt.push_back(tree)
	mode = int(data.get("mode", mode))
	num_trees = int(data.get("num_trees", rebuilt.size()))
	max_depth = int(data.get("max_depth", max_depth))
	min_samples_split = int(data.get("min_samples_split", min_samples_split))
	max_features = int(data.get("max_features", max_features))
	trees = rebuilt
	return true


# === The older names === #
# Every method above used to carry a leading underscore, which in Godot marks a
# method as virtual or private: the engine calls _ready() and _process(), you do not.
# The names below are the ones that shipped, kept working so nothing that already
# calls them breaks. They only forward. Prefer the ones without the underscore.

func _set_seed(value):
	set_seed(value)

func _reset():
	reset()

func _fit(newX, newY):
	return fit(newX, newY)

func _predict(newX):
	return predict(newX)



# === End Random forest === #
