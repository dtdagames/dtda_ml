const PLAN = 29

# a model this reads from end to end, so every call below is wrong in exactly one field: two wrong at once and either guard could be the one answering
func _but(key, value):
	var data = {"model": "DTDAKMeans", "version": 1, "inertia": 1.5, "centroids": [[0.0, 1.0], [5.0, 6.0]], "scaler": {"mode": 0, "offsets": [0.0, 0.0], "scales": [1.0, 1.0]}}
	data[key] = value
	return data

func _seeded(groups, seed_value):
	var model = DTDAKMeans.new(groups)
	model.set_seed(seed_value)
	return model

func _run(t):
	t.section("K-Means (the errors further down are expected)")
	var X = [[0.0, 0.0], [0.4, 0.3], [-0.3, 0.4], [10.0, 0.0], [10.4, 0.3], [9.7, 0.4], [5.0, 9.0], [5.4, 9.3], [4.7, 9.4]]
	var km = _seeded(3, 1)
	var groups = km.fit_predict(X)
	t.check_equal("a centre comes back as a plain Array", typeof(km.centroids[0]), TYPE_ARRAY)
	t.check_near("_fit leaves the inertia of the rows it was given", km.inertia_of(X), km.inertia, 0.0)
	t.check("rows that do not sit on their centre leave an inertia to measure", km.inertia > 0.0)
	var missed = 0
	for s in 40:
		var f = _seeded(3, s + 1).fit_predict(X)
		missed += int(not (f[0] == f[1] and f[1] == f[2] and f[3] == f[4] and f[4] == f[5] and f[6] == f[7] and f[7] == f[8] and f[0] != f[3] and f[3] != f[6] and f[0] != f[6]))
	t.check_equal("forty seeds, forty times the planted grouping", missed, 0)
	# the scaler is inside for this: euclidean distances would let a column counted in thousands drown the column the groups live in. Integer columns on both sides, so a scale truncated to an integer answers here as two different groupings
	var unit = [[1, 0], [2, 3], [1, 9], [2, 12], [1, 20], [2, 23]]
	var thousands = [[1000, 0], [2000, 3], [1000, 9], [2000, 12], [1000, 20], [2000, 23]]
	t.check_equal("a column multiplied by a thousand gives the same answer", _seeded(3, 4).fit_predict(thousands), _seeded(3, 4).fit_predict(unit))
	var flat = _seeded(2, 1)
	var flat_groups = flat.fit_predict([[7, 0], [7, 1], [7, 8], [7, 9]])
	var middles = [minf(flat.get_centroids()[0][1], flat.get_centroids()[1][1]), maxf(flat.get_centroids()[0][1], flat.get_centroids()[1][1])]
	t.check_near_array("a centre sits on the middle of the rows it holds", middles, [0.5, 8.5], 1e-9)
	t.check("and the groups follow the column that changes", flat_groups[0] == flat_groups[1] and flat_groups[2] == flat_groups[3] and flat_groups[0] != flat_groups[2])
	t.check_equal("rows that are all the same put every one of them in the first group", _seeded(3, 1).fit_predict([[5.0, 5.0], [5.0, 5.0], [5.0, 5.0], [5.0, 5.0]]), [0, 0, 0, 0])
	var starved = _seeded(3, 1)
	starved.fit([[0.0], [0.0], [10.0], [10.0]])
	var centres = starved.get_centroids().map(func(c): return c[0])
	centres.sort()
	t.check_near_array("a group left holding nothing keeps the centre it had, and k centres are still owed", centres, [0.0, 0.0, 10.0], 1e-9)
	var repeated = 0
	for i in 200:
		var start = km._initial_centroids(X)
		repeated += int(start.size() != 3 or start.count(start[0]) > 1 or start.count(start[1]) > 1)
	t.check_equal("no draw among two hundred takes the same row twice", repeated, 0)
	var back = DTDAKMeans.new()
	t.check("K-Means _save and _load report a success", km.save("user://dtda_ml_test_kmeans.json") and back.load("user://dtda_ml_test_kmeans.json"))
	t.check_equal("a reloaded model answers the same groups", back.predict(X), groups)
	t.check_near("the inertia comes back", back.inertia, km.inertia, 1e-9)
	t.check_equal("and so do the settings", [back.k, back.max_iterations, back.num_runs], [3, 100, 5])
	t.check_equal("a fit refuses a row holding a nan", km.fit([[1.0, 2.0], [2.0, 1.0], [8.0, NAN]]), false)
	t.check_equal("_fit for no group at all is refused", DTDAKMeans.new(0).fit(X), false)
	t.check_equal("_fit with fewer rows than groups is refused", DTDAKMeans.new(99).fit(X), false)
	t.check_equal("_fit for no run at all is refused", DTDAKMeans.new(3, 100, 0).fit(X), false)
	t.check_equal("a refused fit leaves the model answering as before", km.predict(X), groups)
	t.check_equal("_load refuses a file that lies about its model name", km.from_dict(_but("model", "NotAKMeans")), false)
	t.check_equal("_load refuses another format version", km.from_dict(_but("version", 99)), false)
	t.check_equal("_load refuses a centre that is not a row of numbers", km.from_dict(_but("centroids", [[0.0, 1.0], [0.0, "nope"]])), false)
	t.check_equal("_load refuses centres of different widths", km.from_dict(_but("centroids", [[0.0, 1.0], [0.0]])), false)
	t.check_equal("_load refuses an inertia that is not a number", km.from_dict(_but("inertia", "nope")), false)
	t.check_equal("_load refuses a scaler it cannot use", km.from_dict(_but("scaler", {"mode": 0, "offsets": [0.0, 0.0], "scales": [1.0, 0.0]})), false)
	t.check_equal("_load refuses a scaler as wide as the centres are not", km.from_dict(_but("centroids", [[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]])), false)
	t.check_equal("the seven refusals above were all handed to the model fitted at the top, and it answers as before", km.predict(X), groups)
	km.reset()
	t.check_empty("_reset forgets the centres", km.predict(X))
	km.fit(X)
	t.check_equal("K-Means _reset replays the same draws", km.predict(X), groups)
