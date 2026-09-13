const PLAN = 16

# one feature that matters and five that do not: the label follows x0, and one row in eight carries the wrong one, which is the noise a deep tree learns by heart. Arithmetic rather than a generator, so the rows are the same on every engine whatever its random stream
func _rows(first, count, flip_every):
	var X = []
	var y = []
	for i in range(first, first + count):
		X.push_back([i % 20, (i * 37 + 53) % 13, (i * 37 + 106) % 13, (i * 37 + 159) % 13, (i * 37 + 212) % 13, (i * 37 + 265) % 13])
		y.push_back((1 if i % 20 >= 10 else 0) if flip_every == 0 or i % flip_every != 3 else (0 if i % 20 >= 10 else 1))
	return [X, y]

func _grown(count, depth, forest_mode, features, seed_value, data):
	var forest = DTDAForest.new(count, depth, 2, forest_mode, features)
	forest.set_seed(seed_value)
	forest.fit(data[0], data[1])
	return forest

func _run(t):
	t.section("Random forest (the errors further down are expected)")
	var noisy = _rows(0, 48, 8)
	var held_out = _rows(1000, 48, 0)
	t.check_equal("the square root of the features classifying, a third regressing, floored and never down to none, an explicit count as it is and capped at the count there is", [DTDAForest.new()._resolved_max_features(16), DTDAForest.new(5, 3, 2, DTDAForest.REGRESSOR)._resolved_max_features(16), DTDAForest.new(5, 3, 2, DTDAForest.REGRESSOR)._resolved_max_features(5), DTDAForest.new(5, 3, 2, DTDAForest.REGRESSOR)._resolved_max_features(2), DTDAForest.new(5, 3, 2, DTDAForest.CLASSIFIER, 3)._resolved_max_features(16), DTDAForest.new(5, 3, 2, DTDAForest.CLASSIFIER, 99)._resolved_max_features(6), DTDAForest.new()._resolved_max_features(0)], [4, 5, 1, 1, 3, 6, 1])
	t.check_equal("the majority label wins, a tie goes to the label the first tree answered, and a regressor averages instead of voting", [DTDAForest.new()._combine([0, 1, 1]), DTDAForest.new()._combine([1, 0]), DTDAForest.new(5, 3, 2, DTDAForest.REGRESSOR)._combine([1.0, 2.0, 6.0])], [1, 1, 3.0])
	var bagged = _grown(6, 6, DTDAForest.CLASSIFIER, 99, 3, noisy)
	var thin = _grown(6, 6, DTDAForest.CLASSIFIER, 1, 3, noisy)
	# every feature offered to every split, so nothing but the draw of rows can tell the trees of bagged apart; thin is the same forest but for the count a split may look at, read on the trees themselves rather than guessed from the draw, six columns resolving to two by the rule, to one asked for and to six capped
	t.check_equal("bagging alone already grows different trees", t._same(bagged.trees[0].root, bagged.trees[1].root), false)
	t.check_equal("the count the rule resolves is the count every tree is grown with, and the same bags with one feature per split grow a different tree again", [_grown(2, 3, DTDAForest.CLASSIFIER, 0, 5, noisy).trees[0].max_features, thin.trees[0].max_features, bagged.trees[0].max_features, t._same(thin.trees[0].root, bagged.trees[0].root)], [2, 1, 6, false])
	var lone = DTDATree.new(8, 2, DTDATree.CLASSIFIER)
	lone.fit(noisy[0], noisy[1])
	# 48 held out rows, so a single row is worth 2.08 points and one forest can land level with the tree; the average over five seeds was measured on eight disjoint groups, from +8.34 to +10.84 points, so 3.0 is far below anything seen and far above nothing
	t.check("five forests average well above the lone deep tree, on rows none of them ever saw", DTDATools.new()._mean_array(range(5).map(func(k): return DTDATools.new().accuracy(_grown(25, 8, DTDAForest.CLASSIFIER, 0, k + 1, noisy).predict(held_out[0]), held_out[1]))) - DTDATools.new().accuracy(lone.predict(held_out[0]), held_out[1]) > 3.0)
	var first = bagged.predict(held_out[0])
	t.check_equal("the same seed grows the same forest", _grown(6, 6, DTDAForest.CLASSIFIER, 99, 3, noisy).predict(held_out[0]), first)
	bagged.reset()
	t.check_empty("_reset forgets the trees", bagged.predict(held_out[0]))
	bagged.fit(noisy[0], noisy[1])
	t.check_equal("_reset replays the same draws", bagged.predict(held_out[0]), first)
	t.check_equal("a fit is refused for a row holding a nan, for more rows than labels, for no tree at all and for labels that are not numbers when regressing, and the four of them leave the forest answering as before", [bagged.fit([[1.0, 2.0], [2.0, 1.0], [8.0, NAN]], [0, 1, 1]), bagged.fit([[1.0, 2.0], [2.0, 1.0], [8.0, 9.0]], [0]), DTDAForest.new(0).fit(noisy[0], noisy[1]), DTDAForest.new(3, 3, 2, DTDAForest.REGRESSOR).fit([[1.0], [2.0]], ["red", "blue"]), bagged.predict(held_out[0])], [false, false, false, false, first])
	t.check_equal("classifying, a label that names a class is not a fault and is voted back", _grown(5, 3, DTDAForest.CLASSIFIER, 0, 1, [[[0.0, 0.0], [0.5, 0.5], [9.0, 9.0], [9.5, 9.5]], ["cave", "cave", "camp", "camp"]]).predict([[0.2, 0.2], [9.2, 9.2]]), ["cave", "camp"])
	var reg = _grown(8, 4, DTDAForest.REGRESSOR, 0, 11, [[[1.6], [4.6], [4.2], [4.1], [5.4], [8.1], [8.9], [9.2], [9.3], [10.2]], [40000, 60000, 58000, 59000, 80000, 100000, 110000, 114000, 121000, 121000]])
	var back = DTDAForest.new()
	t.check_near_array("a forest saves, loads and predicts the same", back.predict([[7.2], [9.0], [11.1]]) if reg.save("user://dtda_ml_test_forest.json") and back.load("user://dtda_ml_test_forest.json") else [], reg.predict([[7.2], [9.0], [11.1]]), 0.0)
	t.check_equal("and carries the same trees and the same growth limits", [back.trees.size(), back.mode, back.num_trees, back.max_depth, back.min_samples_split, back.max_features], [8, DTDAForest.REGRESSOR, 8, 4, 2, 0])
	t.check_equal("_load refuses a file that only lies about its model name, and another that only lies about its format version", [DTDAForest.new().from_dict({"model": "NotAForest", "version": 1, "mode": 0, "num_trees": 1, "max_depth": 5, "min_samples_split": 2, "max_features": 0, "trees": [{"model": "DTDATree", "version": 1, "root": {"leaf": 1}}]}), DTDAForest.new().from_dict({"model": "DTDAForest", "version": 99, "mode": 0, "num_trees": 1, "max_depth": 5, "min_samples_split": 2, "max_features": 0, "trees": [{"model": "DTDATree", "version": 1, "root": {"leaf": 1}}]})], [false, false])
	t.check_equal("_load refuses an empty list of trees", DTDAForest.new().from_dict({"model": "DTDAForest", "version": 1, "mode": 0, "num_trees": 1, "max_depth": 5, "min_samples_split": 2, "max_features": 0, "trees": []}), false)
	t.check_equal("_load refuses a tree it cannot read", DTDAForest.new().from_dict({"model": "DTDAForest", "version": 1, "mode": 0, "num_trees": 1, "max_depth": 5, "min_samples_split": 2, "max_features": 0, "trees": [{"model": "DTDATree", "version": 1, "root": "nope"}]}), false)
	t.check_equal("a refused file leaves the forest alone", [reg.from_dict({"model": "DTDAForest", "version": 1, "mode": 0, "num_trees": 1, "max_depth": 5, "min_samples_split": 2, "max_features": 0, "trees": ["nope"]}), reg.predict([[7.2], [9.0], [11.1]])], [false, back.predict([[7.2], [9.0], [11.1]])])
