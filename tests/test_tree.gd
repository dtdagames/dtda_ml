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

	t.section("Decision tree guards (the errors below are expected)")
	t.check_empty("_predict before _fit", DTDATree.new()._predict([[1]]))
	t.check("_save before _fit fails", not DTDATree.new()._save(path))
	# a file written by another model, saved here so this suite stays self contained
	var other = DTDAKNN.new(1)
	other._fit([[0]], [1])
	var other_path = "user://dtda_ml_test_not_a_tree.json"
	other._save(other_path)
	t.check("_load refuses another kind of model", not DTDATree.new()._load(other_path))
