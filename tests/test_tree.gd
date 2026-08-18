# DTDATree, the CART decision tree.

const DATA_LINR = [
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

const DATA_LOGR = [
	[2, 4, 2, 1, 0, 0, 0],
	[2, 2, 4, 0, 0, 0, 0],
	[4, 2, 1, 1, 0, 1, 1],
	[2, 2, 4, 0, 1, 1, 1],
]

# how many assertions this suite runs, checked by the runner
const PLAN = 29

func _run(t):
	var ml = MLTools.new()

	t.section("Decision tree, classification")
	var X_log = ml._dropVariable(DATA_LOGR, 6)
	var y_log = ml._getVariable(DATA_LOGR, 6)
	var tree = DTDATree.new(3, 2, DTDATree.CLASSIFIER)
	tree._fit(X_log, y_log)
	t.check_near_array("separates the training set", tree._predict(X_log), y_log)

	# the canonical non linear case: no single feature helps at the root,
	# yet each half becomes separable one level down
	var xor_X = [[0, 0], [0, 1], [1, 0], [1, 1]]
	var xor_y = [0, 1, 1, 0]
	var xor_tree = DTDATree.new(4, 2, DTDATree.CLASSIFIER)
	xor_tree._fit(xor_X, xor_y)
	t.check_near_array("separates a XOR", xor_tree._predict(xor_X), xor_y)

	t.section("Decision tree, regression")
	var X_lin = ml._dropVariable(DATA_LINR, 1)
	var y_lin = ml._getVariable(DATA_LINR, 1)
	var regressor = DTDATree.new(3, 2, DTDATree.REGRESSOR)
	regressor._fit(X_lin, y_lin)
	t.check("fits the training set closely", ml._r2_score(regressor._predict(X_lin), y_lin) > 0.99)
	# each leaf answers the mean of the rows it holds
	# x=7.2 falls on the leaf holding 8.1 alone, x=11.1 on the leaf holding 10.2 alone,
	# x=9.0 on the leaf holding 8.9, 9.2 and 9.3, whose mean is 334000/3
	t.check_near_array("a leaf answers the mean of its rows",
		regressor._predict([[7.2], [9.0], [11.1]]),
		[100000.0, 334000.0 / 3.0, 121000.0], 0.01)

	t.section("Decision tree, scale independence")
	# the tree compares features to thresholds, so it needs no scaling at all
	var scaled_X = []
	for row in X_lin:
		scaled_X.push_back([row[0] * 1000.0])
	var scaled_tree = DTDATree.new(3, 2, DTDATree.REGRESSOR)
	scaled_tree._fit(scaled_X, y_lin)
	t.check_near_array("features x1000 give the same predictions",
		scaled_tree._predict([[7200.0], [9000.0], [11100.0]]),
		regressor._predict([[7.2], [9.0], [11.1]]), 0.01)

	t.section("Decision tree, growth limits")
	# depth 0 cannot split at all, the whole tree is a single leaf on the majority label
	var stump = DTDATree.new(0, 2, DTDATree.CLASSIFIER)
	stump._fit([[0], [1], [1]], [0, 1, 1])
	t.check_near_array("max_depth 0 answers the majority label", stump._predict([[0], [1]]), [1, 1])
	# min_samples_split larger than the training set forbids any split
	var blocked = DTDATree.new(5, 99, DTDATree.CLASSIFIER)
	blocked._fit([[0], [1]], [0, 1])
	t.check_equal("min_samples_split blocks the split", blocked._predict([[0]]).size(), 1)

	# a threshold is a midpoint between two values of the training set, so no training
	# row ever sits exactly on one and only a prediction row can. The rule is that it
	# goes left, and nothing else in this suite pins it: turning <= into < in
	# _predict_row() leaves every other assertion green
	var boundary = DTDATree.new(1, 2, DTDATree.CLASSIFIER)
	boundary._fit([[0], [2]], [0, 1])
	t.check_near_array("a row sitting on the threshold goes left", boundary._predict([[1.0]]), [0])

	t.section("Decision tree, edge cases")
	var single = DTDATree.new(5, 2, DTDATree.CLASSIFIER)
	single._fit([[3, 4]], [7])
	t.check_near_array("a single training row", single._predict([[9, 9]]), [7])

	# identical rows carrying different labels: no threshold exists, must not loop forever
	var ambiguous = DTDATree.new(5, 2, DTDATree.CLASSIFIER)
	ambiguous._fit([[1, 1], [1, 1]], [0, 1])
	t.check_equal("identical rows with different labels", ambiguous._predict([[1, 1]]).size(), 1)

	# a constant column carries no information, the other one must still be used
	var constant = DTDATree.new(5, 2, DTDATree.CLASSIFIER)
	constant._fit([[5, 1], [5, 2], [5, 3]], [0, 0, 1])
	t.check_near_array("a constant column is ignored", constant._predict([[5, 1], [5, 3]]), [0, 1])

	t.section("Decision tree, how many features a split looks at")
	# max_features is what a forest needs from a tree. 0, the default, is the tree
	# as it always was: every feature, in order, and no draw at all
	var every = DTDATree.new(3, 2, DTDATree.CLASSIFIER)
	every._fit(X_log, y_log)
	t.check_equal("the default looks at every feature in order",
		every._features_for_split(), [0, 1, 2, 3, 4, 5])
	# and it must not even touch the generator, or a tree standing on its own would
	# answer differently depending on what a forest did before it
	every._set_seed(42)
	for i in 5:
		every._features_for_split()
	var untouched = DTDATree.new(3, 2, DTDATree.CLASSIFIER)
	untouched._set_seed(42)
	t.check_equal("the default draw leaves the generator where it was",
		every.rng.randi(), untouched.rng.randi())
	var wide = DTDATree.new(3, 2, DTDATree.CLASSIFIER, 99)
	wide._fit(X_log, y_log)
	t.check_equal("asking for more features than there are draws nothing either",
		wide._features_for_split(), [0, 1, 2, 3, 4, 5])

	var drawing = DTDATree.new(3, 2, DTDATree.CLASSIFIER, 2)
	drawing._fit(X_log, y_log)
	drawing._set_seed(9)
	var drawn = drawing._features_for_split()
	t.check_equal("a draw hands back as many features as asked", drawn.size(), 2)
	t.check_equal("and never the same feature twice", drawn[0] == drawn[1], false)
	t.check("every feature drawn is one that exists",
		drawn[0] >= 0 and drawn[0] < 6 and drawn[1] >= 0 and drawn[1] < 6)
	var same_draw = DTDATree.new(3, 2, DTDATree.CLASSIFIER, 2)
	same_draw._fit(X_log, y_log)
	same_draw._set_seed(9)
	t.check_equal("the same seed draws the same features", same_draw._features_for_split(), drawn)

	t.section("Decision tree, saving and loading")
	var path = "user://dtda_ml_test_tree.json"
	t.check("_save reports a success", tree._save(path))
	var reloaded = DTDATree.new()
	t.check("_load reports a success", reloaded._load(path))
	# the feature index goes through JSON as a float and must come back usable
	t.check_near_array("a reloaded tree predicts the same", reloaded._predict(X_log), tree._predict(X_log))

	var regressor_path = "user://dtda_ml_test_tree_reg.json"
	t.check("the regressor saves", regressor._save(regressor_path))
	var reg_back = DTDATree.new()
	t.check("the regressor loads", reg_back._load(regressor_path))
	t.check_near_array("a reloaded regressor predicts the same",
		reg_back._predict([[7.2], [9.0], [11.1]]), regressor._predict([[7.2], [9.0], [11.1]]), 0.01)

	var drawing_path = "user://dtda_ml_test_tree_draw.json"
	drawing._save(drawing_path)
	var drawing_back = DTDATree.new()
	drawing_back._load(drawing_path)
	t.check_equal("max_features comes back from the file", drawing_back.max_features, 2)

	t.section("Decision tree guards (the errors below are expected)")
	t.check_empty("_predict before _fit", DTDATree.new()._predict([[1]]))
	t.check_equal("_save before _fit fails", DTDATree.new()._save(path), false)
	# a file written by another model, saved here so this suite stays self contained
	var other = DTDAKNN.new(1)
	other._fit([[0]], [1])
	var other_path = "user://dtda_ml_test_not_a_tree.json"
	other._save(other_path)
	t.check_equal("_load refuses another kind of model", DTDATree.new()._load(other_path), false)
	# a model file lives in user://, where it can be edited by hand, and DTDAForest
	# hands whole subtrees straight to _from_dict()
	var handmade = "user://dtda_ml_test_tree_handmade.json"
	var handmade_file = FileAccess.open(handmade, FileAccess.WRITE)
	handmade_file.store_string('{"model": "DTDATree", "version": 1, "root": "not a node"}')
	handmade_file.close()
	t.check_equal("_load refuses a tree whose root is not a node",
		DTDATree.new()._load(handmade), false)
