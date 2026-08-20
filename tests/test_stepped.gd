# Training a slice at a time.
#
# fit() runs to the end before it returns, which on a forest is seconds: a frozen
# frame. The same training can be taken a slice per frame instead. fit() is that loop
# and nothing else, so everything the other suites assert about fit() is already
# asserting about the stepping underneath; what is left to pin is here.

# how many assertions this suite runs, checked by the runner
const PLAN = 29

const X = [[1.0, 2.0], [2.0, 1.0], [8.0, 9.0], [9.0, 8.0], [3.0, 3.0], [7.0, 7.0]]
const Y = [0, 0, 1, 1, 0, 1]
const PROBE = [[2.0, 2.0], [8.5, 8.5]]

func _stepped(model, rows, labels):
	if labels == null:
		if not model.fit_begin(rows):
			return -1
	elif not model.fit_begin(rows, labels):
		return -1
	var steps := 0
	while model.is_fitting():
		model.fit_step()
		steps += 1
	return steps

func _run(t):
	t.section("Stepped training, the same model either way")
	# every one of these takes its slices in a different unit: a round of the descent,
	# a pass over the rows, a run from fresh starts, a whole tree
	for pair in [["LinReg", DTDALinReg.new(0.01, 30), DTDALinReg.new(0.01, 30)],
			["LogReg", DTDALogReg.new(0.01, 30), DTDALogReg.new(0.01, 30)],
			["SVM", DTDASVM.new(0.01, 0.01, 30), DTDASVM.new(0.01, 0.01, 30)]]:
		pair[1].fit(X, Y)
		var steps: int = _stepped(pair[2], X, Y)
		t.check_equal("%s takes one slice per round" % pair[0], steps, 30)
		t.check_near_array("%s steps to the very same model" % pair[0],
			pair[2].predict(PROBE), pair[1].predict(PROBE), 0.0)

	var forest_whole := DTDAForest.new(5, 3, 2, DTDAForest.CLASSIFIER)
	forest_whole.set_seed(3)
	forest_whole.fit(X, Y)
	var forest_stepped := DTDAForest.new(5, 3, 2, DTDAForest.CLASSIFIER)
	forest_stepped.set_seed(3)
	t.check_equal("a forest takes one slice per tree", _stepped(forest_stepped, X, Y), 5)
	t.check_equal("a forest steps to the very same model",
		forest_stepped.predict(PROBE), forest_whole.predict(PROBE))

	var km_whole := DTDAKMeans.new(2, 50, 4)
	km_whole.set_seed(3)
	km_whole.fit(X)
	var km_stepped := DTDAKMeans.new(2, 50, 4)
	km_stepped.set_seed(3)
	t.check_equal("a K-Means takes one slice per run", _stepped(km_stepped, X, null), 4)
	t.check_equal("a K-Means steps to the very same model", km_stepped.predict(X), km_whole.predict(X))
	t.check_near("with the same inertia", km_stepped.inertia, km_whole.inertia, 0.0)

	t.section("Stepped training, where it is up to")
	var walker := DTDALinReg.new(0.01, 10)
	t.check_equal("nothing under way to begin with", walker.is_fitting(), false)
	t.check_equal("fit_begin answers true when it took the rows", walker.fit_begin(X, Y), true)
	t.check_equal("and then there is something under way", walker.is_fitting(), true)
	var first: float = walker.fit_step()
	t.check("the first slice reports part of the way", first > 0.0 and first < 1.0)
	var last: float = 0.0
	while walker.is_fitting():
		last = walker.fit_step()
	t.check_near("the last slice reports the whole way", last, 1.0, 0.0)
	t.check_equal("and nothing is under way any more", walker.is_fitting(), false)

	t.section("Stepped training, a training that never finishes (errors below are expected)")
	# The invariant that governs a refused fit and a refused file, applied to time:
	# either the model it was going to replace, whole, or the new one, whole, and
	# never a half of each. Three ways of not finishing, so it does not rest on one
	var standing := DTDALinReg.new(0.01, 40)
	standing.fit(X, Y)
	var before := standing.predict(PROBE)
	# one: begun on other data, stepped part of the way, then simply abandoned
	standing.fit_begin([[9.0, 9.0], [1.0, 1.0], [5.0, 5.0]], [50.0, 60.0, 70.0])
	standing.fit_step()
	standing.fit_step()
	t.check_near_array("abandoned halfway, it answers what it answered before",
		standing.predict(PROBE), before, 0.0)
	# two: cancelled outright
	standing.fit_cancel()
	t.check_equal("cancelled, nothing is under way", standing.is_fitting(), false)
	t.check_near_array("and it still answers what it answered before",
		standing.predict(PROBE), before, 0.0)
	# three: begun on rows it refuses, which never starts anything
	t.check_equal("fit_begin answers false on rows it cannot use",
		standing.fit_begin([[1.0, 2.0], "nope"], [0, 1]), false)
	t.check_equal("and starts nothing", standing.is_fitting(), false)
	t.check_near_array("the model being what it always was",
		standing.predict(PROBE), before, 0.0)

	# and a forest, whose slices are whole trees
	var grove := DTDAForest.new(6, 3, 2, DTDAForest.CLASSIFIER)
	grove.set_seed(5)
	grove.fit(X, Y)
	var grove_before := grove.predict(PROBE)
	grove.fit_begin([[9.0, 9.0], [1.0, 1.0], [5.0, 5.0]], [1, 0, 1])
	grove.fit_step()
	t.check_equal("a forest abandoned after one tree keeps the trees it had",
		grove.predict(PROBE), grove_before)
	t.check_equal("and still holds the number of trees it grew", grove.trees.size(), 6)

	t.section("Stepped training, asking for a slice when there is none")
	t.check_near("a step with no training under way answers the whole way",
		DTDALinReg.new(0.01, 10).fit_step(), 1.0, 0.0)
	# a model that trains in one go says so rather than looping for ever
	t.check_near("a KNN has no slices to take", DTDAKNN.new(1).fit_step(), 1.0, 0.0)
	t.check_near("nor has a lone tree", DTDATree.new(3, 2, DTDATree.CLASSIFIER).fit_step(), 1.0, 0.0)
	t.check_equal("and neither is left thinking it is training",
		DTDAKNN.new(1).is_fitting(), false)
