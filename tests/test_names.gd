# The names the addon answers to.
#
# Every method a user calls used to carry a leading underscore, which in Godot marks
# a method as virtual or private. They have lost it, and the older names are kept as
# wrappers so that nothing already written breaks. This suite drives every one of
# those wrappers: the day one is dropped by accident, it says so.

# how many assertions this suite runs, checked by the runner
const PLAN = 55

const X = [[1.0, 2.0], [2.0, 1.0], [8.0, 9.0], [9.0, 8.0], [3.0, 3.0], [7.0, 7.0]]
const Y = [0, 0, 1, 1, 0, 1]
const PROBE = [[2.0, 2.0], [8.5, 8.5]]

# a model extended the way the compatibility notes tell people to extend one
class Overrider extends DTDAKMeans:
	func predict(newX):
		return ["the override answered"]

func _run(t):
	var ml = DTDATools.new()

	t.section("Names, the toolbox itself")
	# MLTools was generic enough to collide with the next addon that has one
	t.check("DTDATools is what the toolbox is called now", DTDATools.new() != null)
	t.check_near("the older name still carries the toolbox", MLTools.new().accuracy([1, 1], [1, 1]), 100.0)
	t.check("a model is a DTDATools", DTDAKNN.new(1) is DTDATools)
	# and the one thing the older name cannot keep: the models extend DTDATools, not
	# the subclass that carries the old name. This is here so the README can say it
	# through a box, because the parser settles this one on its own otherwise and
	# refuses to compile the comparison at all, which is a firmer answer than false
	var boxed = [DTDAKNN.new(1)]
	t.check_equal("a model is no longer an MLTools", boxed[0] is MLTools, false)

	t.section("Names, the toolbox methods")
	t.check_near("_get_perf", ml._get_perf([1, 0], [1, 0], 0), ml.get_perf([1, 0], [1, 0], 0))
	t.check_equal("_dropVariable", ml._dropVariable([[1, 2], [3, 4]], 1), ml.drop_variable([[1, 2], [3, 4]], 1))
	t.check_equal("_getVariable", ml._getVariable([[1, 2], [3, 4]], 1), ml.get_variable([[1, 2], [3, 4]], 1))
	t.check_near("_accuracy", ml._accuracy([1, 0], [1, 1]), ml.accuracy([1, 0], [1, 1]))
	t.check_equal("_confusion_matrix", ml._confusion_matrix([1, 0], [1, 1]), ml.confusion_matrix([1, 0], [1, 1]))
	t.check_near("_precision", ml._precision([1, 0], [1, 1]), ml.precision([1, 0], [1, 1]))
	t.check_near("_recall", ml._recall([1, 0], [1, 1]), ml.recall([1, 0], [1, 1]))
	t.check_near("_f1_score", ml._f1_score([1, 0], [1, 1]), ml.f1_score([1, 0], [1, 1]))
	t.check_near("_mse", ml._mse([1.0, 0.0], [1.0, 1.0]), ml.mse([1.0, 0.0], [1.0, 1.0]))
	t.check_near("_rmse", ml._rmse([1.0, 0.0], [1.0, 1.0]), ml.rmse([1.0, 0.0], [1.0, 1.0]))
	t.check_near("_mae", ml._mae([1.0, 0.0], [1.0, 1.0]), ml.mae([1.0, 0.0], [1.0, 1.0]))
	t.check_near("_r2_score", ml._r2_score([1.0, 2.0], [1.0, 3.0]), ml.r2_score([1.0, 2.0], [1.0, 3.0]))

	t.section("Names, the scaler")
	var scaler = DTDAScaler.new()
	scaler._fit(X)
	t.check("_fit on a scaler leaves it fitted", scaler.offsets != null)
	t.check_equal("_transform", scaler._transform(PROBE), scaler.transform(PROBE))
	var other = DTDAScaler.new()
	t.check_equal("_fit_transform", other._fit_transform(X), scaler.transform(X))
	t.check_equal("_inverse_transform", scaler._inverse_transform([[0.0, 0.0]]),
		scaler.inverse_transform([[0.0, 0.0]]))

	t.section("Names, fitting and predicting")
	# every model, through the older names alone, has to end up where the new ones do
	# named one by one rather than in a bare loop, so a failure says which model
	for pair in [["KNN", DTDAKNN.new(1)], ["LinReg", DTDALinReg.new(0.01, 50)],
			["LogReg", DTDALogReg.new(0.01, 50)], ["SVM", DTDASVM.new(0.01, 0.01, 50)],
			["Tree", DTDATree.new(3, 2, DTDATree.CLASSIFIER)]]:
		t.check_equal("%s _fit still fits" % pair[0], pair[1]._fit(X, Y), true)
		t.check_equal("%s _predict answers what predict answers" % pair[0],
			pair[1]._predict(PROBE), pair[1].predict(PROBE))
	# the seed is read straight off the model rather than through an outcome: on six
	# tidy rows every seed grows the same forest, so an outcome would notice nothing
	var forest = DTDAForest.new(3, 3, 2, DTDAForest.CLASSIFIER)
	forest._set_seed(4)
	t.check_equal("_set_seed on a forest", forest.start_seed, 4)
	t.check_equal("_fit on a forest", forest._fit(X, Y), true)
	t.check_equal("_predict on a forest", forest._predict(PROBE), forest.predict(PROBE))
	forest._reset()
	t.check_empty("_reset on a forest", forest.predict(PROBE))

	# a tree only draws features when max_features asks it to, so that is where its
	# seed shows. One of two features, twenty times over: a generator that was never
	# seeded has no way of matching that
	var drawer = DTDATree.new(3, 2, DTDATree.CLASSIFIER, 1)
	drawer.fit(X, Y)
	drawer._set_seed(9)
	var twin_tree = DTDATree.new(3, 2, DTDATree.CLASSIFIER, 1)
	twin_tree.fit(X, Y)
	twin_tree.set_seed(9)
	var drawn = []
	var twin_drawn = []
	for i in 20:
		drawn.append_array(drawer._features_for_split())
		twin_drawn.append_array(twin_tree._features_for_split())
	t.check_equal("_set_seed on a tree", drawn, twin_drawn)

	# What the notes in dtda_ml_tools_compat.gd promise: override predict() and the
	# library reaches your override. It holds because fit_predict() calls predict()
	# by name rather than doing the work itself, and that is a property of this code,
	# not of the language.
	#
	# Do not drop this on the strength of the obvious mutation. Making fit_predict()
	# call _predict() instead leaves it green, the older name forwarding to the
	# override anyway, and it is tempting to conclude the assertion proves nothing.
	# The mutation that shows it is inlining the body of predict() into fit_predict(),
	# which saves a _check_fitted() and a dispatch the caller has just made redundant,
	# and is exactly equivalent for everyone who does not subclass. That is the shape
	# of the regression worth catching: invisible everywhere except where it breaks.
	var overridden = Overrider.new()
	overridden.set_seed(1)
	t.check_equal("the library reaches an override of predict, rather than inlining it",
		overridden.fit_predict(X), ["the override answered"])

	t.section("Names, K-Means")
	var km = DTDAKMeans.new(2, 50, 2)
	km._set_seed(1)
	t.check_equal("_set_seed on a K-Means", km.start_seed, 1)
	t.check_equal("_fit on a K-Means", km._fit(X), true)
	t.check_equal("_predict on a K-Means", km._predict(X), km.predict(X))
	t.check_equal("_fit_predict", km._fit_predict(X), km.predict(X))
	t.check_near("_inertia_of", km._inertia_of(X), km.inertia_of(X), 0.0)
	t.check_near_array("_get_centroids", km._get_centroids()[0], km.get_centroids()[0], 0.0)
	km._reset()
	t.check_empty("_reset on a K-Means", km.predict(X))

	t.section("Names, Q-Learning")
	var agent = DTDAQLearning.new(0.5, 0.9, 0.0)
	agent._set_seed(2)
	t.check_equal("_set_seed on an agent", agent.start_seed, 2)
	t.check_near("_learn", agent._learn("room", "north", 4, "end", [], true), 2.0)
	t.check_near("_get_q", agent._get_q("room", "north"), agent.get_q("room", "north"), 0.0)
	t.check_equal("_choose_action", agent._choose_action("room", ["north", "south"]),
		agent.choose_action("room", ["north", "south"]))
	t.check_equal("_predict on an agent", agent._predict("room"), agent.predict("room"))
	t.check_near("_decay_exploration", agent._decay_exploration(), agent.decay_exploration())
	agent._reset()
	t.check("_reset on an agent", agent.q_table == null)

	t.section("Names, saving and loading")
	var path = "user://dtda_ml_test_names.json"
	var saver = DTDAKNN.new(1)
	saver.fit(X, Y)
	t.check_equal("_save", saver._save(path), true)
	var back = DTDAKNN.new(1)
	t.check_equal("_load", back._load(path), true)
	t.check_equal("a model read through the older name predicts the same",
		back.predict(PROBE), saver.predict(PROBE))
	t.check_equal("_to_dict", back._to_dict()["model"], "DTDAKNN")
	t.check_equal("_from_dict", DTDAKNN.new(1)._from_dict(saver.to_dict()), true)
