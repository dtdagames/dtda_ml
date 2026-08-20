# DTDAForest, the random forest.

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
const PLAN = 54

# A noisy world with one feature that matters and five that do not.
# The label follows x0 alone, and one row in eight carries the wrong one: that is the
# noise a deep tree memorises. Built by arithmetic and not by a generator, so the data
# is the same on every engine, whatever its random stream or its number formatting.
const NOISE_FEATURES = 5

func _rows(first, count, flip_every):
	var X = []
	var y = []
	for k in count:
		var i = first + k
		var x0 = i % 20
		var row = [x0]
		for j in NOISE_FEATURES:
			row.push_back((i * 37 + (j + 1) * 53) % 13)
		var label = 1 if x0 >= 10 else 0
		# flip_every 0 asks for the honest labels, the rule with no noise at all
		if flip_every > 0 and i % flip_every == 3:
			label = 1 - label
		X.push_back(row)
		y.push_back(label)
	return [X, y]

# write a handmade file and try to load it, for the malformed file guards
func _load_written(content, forest = null):
	var path = "user://dtda_ml_test_forest_handmade.json"
	var file = FileAccess.open(path, FileAccess.WRITE)
	file.store_string(content)
	file.close()
	if forest == null:
		forest = DTDAForest.new()
	return forest.load(path)

func _run(t):
	var ml = DTDATools.new()
	var X_log = ml.drop_variable(DATA_LOGR, 6)
	var y_log = ml.get_variable(DATA_LOGR, 6)
	var X_lin = ml.drop_variable(DATA_LINR, 1)
	var y_lin = ml.get_variable(DATA_LINR, 1)

	t.section("Random forest, classification")
	var forest = DTDAForest.new(11, 4, 2, DTDAForest.CLASSIFIER)
	forest.set_seed(7)
	forest.fit(X_log, y_log)
	t.check_near_array("separates the training set", forest.predict(X_log), y_log)
	t.check_near_array("predicts the expected classes", forest.predict([
		[1, 3, 1, 0, 1, 0],
		[2, 2, 4, 1, 1, 1],
		[4, 1, 1, 0, 1, 0],
	]), [0, 1, 1])
	t.check_equal("it grew the trees it was asked for", forest.trees.size(), 11)

	t.section("Random forest, regression")
	var regressor = DTDAForest.new(15, 4, 2, DTDAForest.REGRESSOR)
	regressor.set_seed(11)
	regressor.fit(X_lin, y_lin)
	t.check("follows the training set", ml.r2_score(regressor.predict(X_lin), y_lin) > 0.9)
	var salary = regressor.predict([[9.0]])[0]
	t.check("answers in the unit of the target", salary > 40000.0 and salary < 130000.0)
	# no tolerance: _combine() sums the very same numbers in the very same order
	var each = []
	for tree in regressor.trees:
		each.push_back(tree.predict([[9.0]])[0])
	t.check_near("a forest answers the mean of its trees", salary, ml._mean_array(each), 0.0)

	t.section("Random forest, how the trees are put back together")
	var voter = DTDAForest.new(3, 2, 2, DTDAForest.CLASSIFIER)
	t.check_equal("the majority label wins", voter._combine([0, 1, 1]), 1)
	t.check_equal("a lone dissenter loses", voter._combine([5, 5, 5, 2]), 5)
	# a tie has to break somewhere, and it breaks on the first tree, so the same
	# forest asked the same question twice answers the same thing twice
	t.check_equal("a tie goes to the first tree", voter._combine([1, 0]), 1)
	var averager = DTDAForest.new(3, 2, 2, DTDAForest.REGRESSOR)
	t.check_near("the regressor averages instead of voting", averager._combine([1.0, 2.0, 6.0]), 3.0)

	t.section("Random forest, bagging")
	var noisy = _rows(0, 48, 8)
	# every feature offered to every split, so nothing but the draw of rows can tell
	# these trees apart. They still differ, which is bagging doing its work
	var bagged = DTDAForest.new(6, 6, 2, DTDAForest.CLASSIFIER, 99)
	bagged.set_seed(3)
	bagged.fit(noisy[0], noisy[1])
	var all_identical = true
	for i in range(1, bagged.trees.size()):
		if not t._same(bagged.trees[i].root, bagged.trees[0].root):
			all_identical = false
	t.check_equal("bagging alone already grows different trees", all_identical, false)
	t.check_equal("asking for more features than there are is capped",
		bagged._resolved_max_features(6), 6)

	t.section("Random forest, how many features a split may look at")
	var classifier = DTDAForest.new(5, 3, 2, DTDAForest.CLASSIFIER)
	var regressor_rule = DTDAForest.new(5, 3, 2, DTDAForest.REGRESSOR)
	# the usual rule: the square root when classifying, a third when regressing.
	# 16 features tells the two apart, 4 against 5
	t.check_equal("a classifier looks at the square root of them", classifier._resolved_max_features(16), 4)
	t.check_equal("a regressor looks at a third of them", regressor_rule._resolved_max_features(16), 5)
	# floored, and never down to nothing whatever the count
	t.check_equal("a third of five is one", regressor_rule._resolved_max_features(5), 1)
	t.check_equal("never fewer than one feature", regressor_rule._resolved_max_features(2), 1)
	t.check_equal("never fewer than one, classifying either", classifier._resolved_max_features(1), 1)
	var forced = DTDAForest.new(5, 3, 2, DTDAForest.CLASSIFIER, 3)
	t.check_equal("an explicit count is taken as it is", forced._resolved_max_features(16), 3)

	t.section("Random forest, determinism")
	var twin_a = DTDAForest.new(7, 4, 2, DTDAForest.CLASSIFIER)
	twin_a.set_seed(123)
	twin_a.fit(noisy[0], noisy[1])
	var twin_b = DTDAForest.new(7, 4, 2, DTDAForest.CLASSIFIER)
	twin_b.set_seed(123)
	twin_b.fit(noisy[0], noisy[1])
	var held_out = _rows(1000, 48, 0)
	t.check_equal("the same seed grows the same forest",
		twin_b.predict(held_out[0]), twin_a.predict(held_out[0]))
	# without this a forest could not be replayed twice in a row from one seed
	var before_reset = twin_a.predict(held_out[0])
	twin_a.reset()
	t.check_empty("_reset forgets the trees", twin_a.predict(held_out[0]))
	twin_a.fit(noisy[0], noisy[1])
	t.check_equal("_reset replays the same draws", twin_a.predict(held_out[0]), before_reset)

	t.section("Random forest, generalisation")
	# a deep tree on noisy labels learns the noise by heart: it answers every training
	# row right, including the ones whose label is wrong, and pays for it on rows it
	# has never seen
	var lone = DTDATree.new(8, 2, DTDATree.CLASSIFIER)
	lone.fit(noisy[0], noisy[1])
	var lone_train = ml.accuracy(lone.predict(noisy[0]), noisy[1])
	var lone_test = ml.accuracy(lone.predict(held_out[0]), held_out[1])
	t.check_near("a deep tree memorises its training set", lone_train, 100.0)
	t.check("and does worse on rows it never saw", lone_test < lone_train)
	# averaged over five seeds, not measured on one: this test set holds 48 rows, so a
	# single row is worth 2.08 points and one forest can land level with the tree.
	# the average over five was measured on eight disjoint groups of seeds, from +8.34
	# to +10.84 points, so 3.0 sits far below anything seen and far above nothing
	var total = 0.0
	for k in 5:
		var trial = DTDAForest.new(25, 8, 2, DTDAForest.CLASSIFIER)
		trial.set_seed(k + 1)
		trial.fit(noisy[0], noisy[1])
		total += ml.accuracy(trial.predict(held_out[0]), held_out[1])
	var forest_test = total / 5.0
	t.check("five forests average well above the lone tree", forest_test - lone_test > 3.0)

	t.section("Random forest, a fit that is refused changes nothing")
	var zero = 0.0
	var nan_row = [[1.0, 2.0], [2.0, 1.0], [8.0, zero / zero]]
	var inf_row = [[1.0, 2.0], [2.0, 1.0], [8.0, 1.0 / zero]]
	var text_row = [[1.0, 2.0], [2.0, 1.0], [8.0, "nope"]]
	var ragged = [[1.0, 2.0], [2.0], [8.0, 9.0]]
	var three = [0, 1, 1]
	var sound_rows = [[1.0, 2.0], [2.0, 1.0], [8.0, 9.0]]
	var steady = DTDAForest.new(5, 3, 2, DTDAForest.CLASSIFIER)
	steady.set_seed(2)
	steady.fit(X_log, y_log)
	var steady_before = steady.predict(X_log)
	t.check_equal("a forest refuses a row holding a nan", steady.fit(nan_row, three), false)
	t.check_equal("a forest refuses more rows than labels", steady.fit(sound_rows, [0]), false)
	steady.fit(inf_row, three)
	steady.fit(text_row, three)
	steady.fit(ragged, three)
	t.check_near_array("a forest predicts what it predicted before those four",
		steady.predict(X_log), steady_before)
	t.check_equal("a regressor forest refuses labels that are not numbers",
		DTDAForest.new(3, 3, 2, DTDAForest.REGRESSOR).fit([[1.0], [2.0]], ["red", "blue"]), false)
	# and the other side of that line: classifying, a forest only counts labels and
	# votes among them, so a label naming a class is not a fault
	var named = DTDAForest.new(5, 3, 2, DTDAForest.CLASSIFIER)
	named.set_seed(1)
	t.check_equal("a classifier forest takes labels that name a class",
		named.fit([[0.0, 0.0], [0.5, 0.5], [9.0, 9.0], [9.5, 9.5]], ["cave", "cave", "camp", "camp"]), true)
	t.check_equal("and votes a name back", named.predict([[0.2, 0.2], [9.2, 9.2]]), ["cave", "camp"])

	t.section("Random forest, saving and loading")
	var path = "user://dtda_ml_test_forest.json"
	t.check("_save reports a success", forest.save(path))
	var back = DTDAForest.new()
	t.check("_load reports a success", back.load(path))
	t.check_near_array("a reloaded forest predicts the same",
		back.predict(X_log), forest.predict(X_log))
	t.check_equal("it carries the same trees", back.trees.size(), forest.trees.size())
	t.check_equal("and the same growth limits",
		[back.mode, back.num_trees, back.max_depth, back.min_samples_split, back.max_features],
		[forest.mode, forest.num_trees, forest.max_depth, forest.min_samples_split, forest.max_features])
	var reg_path = "user://dtda_ml_test_forest_reg.json"
	t.check("the regressor saves", regressor.save(reg_path))
	var reg_back = DTDAForest.new()
	t.check("the regressor loads", reg_back.load(reg_path))
	t.check_near_array("a reloaded regressor predicts the same",
		reg_back.predict([[7.2], [9.0], [11.1]]), regressor.predict([[7.2], [9.0], [11.1]]), 0.0)

	t.section("Random forest guards (the errors below are expected)")
	t.check_empty("_predict before _fit", DTDAForest.new().predict([[1]]))
	t.check_equal("_save before _fit fails", DTDAForest.new().save(path), false)
	var empty_fit = DTDAForest.new()
	empty_fit.fit([], [])
	t.check_empty("_fit with no data leaves it unfitted", empty_fit.predict([[1]]))
	var mismatched = DTDAForest.new()
	mismatched.fit([[1], [2]], [1])
	t.check_empty("_fit with fewer labels than rows leaves it unfitted", mismatched.predict([[1]]))
	var treeless = DTDAForest.new(0)
	treeless.fit(X_log, y_log)
	t.check_empty("_fit for no tree at all leaves it unfitted", treeless.predict([[1]]))

	# a file written by another model, saved here so this suite stays self contained
	var other = DTDAKNN.new(1)
	other.fit([[0]], [1])
	var other_path = "user://dtda_ml_test_not_a_forest.json"
	other.save(other_path)
	t.check_equal("_load refuses another kind of model", DTDAForest.new().load(other_path), false)
	t.check_equal("_load refuses a missing file",
		DTDAForest.new().load("user://no_such_forest.json"), false)
	# check_equal against false, not "not <call>": a call that raises answers null,
	# and "not null" is true, which would turn a crash into a pass
	# a readable forest in every respect but its version, so nothing else can answer
	# for the version check
	t.check_equal("_load refuses another format version",
		_load_written('{"model": "DTDAForest", "version": 99, "trees": [{"model": "DTDATree", "version": 1, "root": {"leaf": 1}}]}'), false)
	t.check_equal("_load refuses a file with no trees at all",
		_load_written('{"model": "DTDAForest", "version": 1}'), false)
	# a number rather than a dictionary of trees: a dictionary would be caught further
	# down when its entries turn out not to be trees, and would prove nothing here
	t.check_equal("_load refuses trees that are not a list",
		_load_written('{"model": "DTDAForest", "version": 1, "trees": 5}'), false)
	t.check_equal("_load refuses an empty list of trees",
		_load_written('{"model": "DTDAForest", "version": 1, "trees": []}'), false)
	t.check_equal("_load refuses a tree that is not a dictionary",
		_load_written('{"model": "DTDAForest", "version": 1, "trees": ["nope"]}'), false)
	# the forest hands each entry to DTDATree, whose own guards answer for it
	t.check_equal("_load refuses a tree whose root is not a node",
		_load_written('{"model": "DTDAForest", "version": 1, "trees": [{"model": "DTDATree", "version": 1, "root": "nope"}]}'), false)
	# a file a forest could read from end to end, wrong on the "model" field alone:
	# the DTDAKNN file above is turned away by the guards on the structure
	t.check_equal("DTDAForest refuses a file that only lies about its model name",
		_load_written('{"model": "NotAForest", "version": 1, "mode": 0, "num_trees": 1, "max_depth": 5, "min_samples_split": 2, "max_features": 0, "trees": [{"model": "DTDATree", "version": 1, "root": {"leaf": 1}}]}'), false)
	t.check_equal("_load refuses a list holding something that is not a tree",
		_load_written('{"model": "DTDAForest", "version": 1, "trees": [{"model": "DTDAKNN", "version": 1}]}'), false)
	# a refused file must not take the standing forest down with it
	var survivor = DTDAForest.new(5, 3, 2, DTDAForest.CLASSIFIER)
	survivor.set_seed(4)
	survivor.fit(X_log, y_log)
	var survivor_before = survivor.predict(X_log)
	_load_written('{"model": "DTDAForest", "version": 1, "trees": ["nope"]}', survivor)
	t.check_near_array("a refused file leaves the forest alone", survivor.predict(X_log), survivor_before)
