const PLAN = 25
const PATH = "user://dtda_ml_test_tree.json"

# a file DTDATree reads from end to end, so every call below is wrong in exactly one field: two wrong at once and either guard could be the one answering
func _but(key, value):
	var data = {"model": "DTDATree", "version": 1, "mode": 0, "max_depth": 99, "min_samples_split": 77, "max_features": 0, "root": {"leaf": 1}}
	data[key] = value
	return data

func _grown(depth, min_split, tree_mode, rows, labels, features = 0, seed_value = 0):
	var tree = DTDATree.new(depth, min_split, tree_mode, features)
	tree.set_seed(seed_value)
	tree.fit(rows, labels)
	return tree

func _run(t):
	t.section("Decision tree (the errors further down are expected)")
	var X = [[2, 4, 2, 1, 0, 0], [2, 2, 4, 0, 0, 0], [4, 2, 1, 1, 0, 1], [2, 2, 4, 0, 1, 1]]
	var y = [0, 0, 1, 1]
	var tree = _grown(3, 2, DTDATree.CLASSIFIER, X, y)
	t.check_equal("separates the training set", tree.predict(X), y)
	t.check_equal("separates a XOR, where no feature helps at the root and a gain of zero is still worth taking", _grown(4, 2, DTDATree.CLASSIFIER, [[0, 0], [0, 1], [1, 0], [1, 1]], [0, 1, 1, 0]).predict([[0, 0], [0, 1], [1, 0], [1, 1]]), [0, 1, 1, 0])
	t.check_near_array("a leaf answers the mean of the rows it holds, and integer labels do not truncate it", _grown(3, 2, DTDATree.REGRESSOR, [[1.6], [4.6], [4.2], [4.1], [5.4], [8.1], [8.9], [9.2], [9.3], [10.2]], [40000, 60000, 58000, 59000, 80000, 100000, 110000, 110000, 114000, 121000]).predict([[7.2], [9.0], [11.1]]), [100000.0, 334000.0 / 3.0, 121000.0], 0.01)
	# an integer column, rows out of order, and no pair of neighbours averaging 4.5: the best cut is 4.5 and leaves both sides mixed, while an unsorted column, sides weighed by nothing, an impurity counted in integers or a midpoint divided by an integer all answer something else
	var cut_tree = _grown(1, 2, DTDATree.CLASSIFIER, [[0], [1], [2], [3], [4], [6], [5], [7], [9], [8]], [0, 0, 1, 0, 0, 1, 1, 0, 1, 1])
	t.check_near("the impurity takes the cut that splits best, not the one that peels a pure row off an end", cut_tree.root["threshold"], 4.5, 0.0)
	t.check("max_depth 1 leaves nothing but leaves one level down", cut_tree.root["left"].has("leaf") and cut_tree.root["right"].has("leaf"))
	t.check_equal("max_depth 0 answers the majority label", _grown(0, 2, DTDATree.CLASSIFIER, [[0], [1], [1]], [0, 1, 1]).predict([[0], [1]]), [1, 1])
	t.check_equal("min_samples_split blocks the split, so the root answers the majority label", _grown(5, 99, DTDATree.CLASSIFIER, [[0], [1], [1]], [0, 1, 1]).predict([[0]]), [1])
	t.check_equal("a row sitting on the threshold goes left", _grown(1, 2, DTDATree.CLASSIFIER, [[0], [2]], [0, 1]).predict([[1.0]]), [0])
	t.check("labels that already agree leave the root a leaf, with nothing to lower", _grown(5, 2, DTDATree.CLASSIFIER, [[0], [1], [2]], [1, 1, 1]).root.has("leaf"))
	t.check_equal("identical rows carrying different labels still answer one of them", _grown(5, 2, DTDATree.CLASSIFIER, [[1, 1], [1, 1]], [0, 1]).predict([[1, 1]]).size(), 1)
	t.check_equal("the default looks at every feature in order", tree._features_for_split(), [0, 1, 2, 3, 4, 5])
	t.check_equal("asking for more features than there are draws nothing either", _grown(3, 2, DTDATree.CLASSIFIER, X, y, 99)._features_for_split(), [0, 1, 2, 3, 4, 5])
	var drawing = _grown(3, 2, DTDATree.CLASSIFIER, X, y, 2, 9)
	t.check_equal("the same seed draws the same features", _grown(3, 2, DTDATree.CLASSIFIER, X, y, 2, 9)._features_for_split(), drawing._features_for_split())
	var draws = range(200).map(func(i): return drawing._features_for_split())
	t.check_equal("no draw among two hundred repeats a feature, hands back the wrong number, or names one that does not exist", draws.filter(func(s): return s.size() != 2 or s[0] == s[1] or s.min() < 0 or s.max() >= 6).size(), 0)
	t.check_equal("a tree refuses a row holding a nan", tree.fit([[1.0, 2.0], [2.0, 1.0], [8.0, NAN]], [0, 1, 1]), false)
	t.check_equal("a tree refuses more rows than labels", tree.fit([[1.0, 2.0], [2.0, 1.0]], [0]), false)
	t.check_equal("a refused fit leaves the tree answering as before", tree.predict(X), y)
	t.check_equal("a regressor tree refuses labels that are not numbers", DTDATree.new(3, 2, DTDATree.REGRESSOR).fit([[1.0], [2.0]], ["red", "blue"]), false)
	var back = DTDATree.new()
	t.check_equal("a classifier takes a label that names a class, saves, loads, and hands the name and max_features back", [_grown(3, 2, DTDATree.CLASSIFIER, [[1.0], [2.0]], ["red", "blue"], 2).save(PATH) and back.load(PATH), back.predict([[1.0], [2.0]]), back.max_features], [true, ["red", "blue"], 2])
	t.check_empty("_predict before _fit", DTDATree.new().predict([[1]]))
	t.check_equal("_save before _fit fails", DTDATree.new().save(PATH), false)
	t.check_equal("DTDATree refuses a file that only lies about its model name", DTDATree.new().from_dict(_but("model", "NotATree")), false)
	# a model file lives in user://, where it can be edited by hand, and DTDAForest hands whole subtrees straight to from_dict()
	t.check_equal("_load refuses a tree whose root is not a node", tree.from_dict(_but("root", "not a node")), false)
	t.check_equal("a refused file leaves the growth limits alone", [tree.max_depth, tree.min_samples_split], [3, 2])
	t.check_equal("and leaves the tree predicting as before", tree.predict(X), y)
