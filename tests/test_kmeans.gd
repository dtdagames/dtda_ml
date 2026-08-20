# DTDAKMeans, the k-means clustering.

# how many assertions this suite runs, checked by the runner
const PLAN = 56

# a file this model reads from end to end, which every guard below breaks in exactly
# one place: two fields wrong at once and either guard could be the one answering
const SOUND = '{"model": "DTDAKMeans", "version": 1, "k": 2, "max_iterations": 100, "num_runs": 5, "inertia": 1.5, "centroids": [[0.0, 1.0], [5.0, 6.0]], "scaler": {"mode": 0, "offsets": [0.0, 0.0], "scales": [1.0, 1.0]}}'
const CENTRES_FIELD = '"centroids": [[0.0, 1.0], [5.0, 6.0]]'
const SCALER_FIELD = '"scaler": {"mode": 0, "offsets": [0.0, 0.0], "scales": [1.0, 1.0]}'

# Three blobs, far apart, built by arithmetic and not by a generator, so the data is
# the same on every engine whatever its random stream or its number formatting.
const BLOB_CENTRES = [[0.0, 0.0], [10.0, 0.0], [5.0, 9.0]]

func _blobs():
	var X = []
	var planted = []
	for b in BLOB_CENTRES.size():
		for j in 8:
			var i = b * 8 + j
			X.push_back([BLOB_CENTRES[b][0] + ((i * 7) % 5) * 0.3 - 0.6,
				BLOB_CENTRES[b][1] + ((i * 11) % 5) * 0.3 - 0.6])
			planted.push_back(b)
	return [X, planted]

# four blobs strung along a line, close enough that a single run often settles on a
# grouping that is not the best one. This is where several runs earn their keep
func _line():
	var X = []
	var centres = [0.0, 3.0, 6.0, 9.0]
	for b in 4:
		for j in 6:
			var i = b * 6 + j
			X.push_back([centres[b] + ((i * 7) % 5) * 0.4 - 0.8, ((i * 11) % 5) * 0.4 - 0.8])
	return X

# column 0 says nothing about the groups and is written in whatever unit is asked for,
# column 1 is where the groups actually live
func _mixed(factor):
	var X = []
	for i in 18:
		X.push_back([(1.0 + float(i % 2)) * factor, float((i % 3) * 5) + ((i * 7) % 3) * 0.1])
	return X

# the same grouping, whatever number each group happens to have been given: k-means
# has no reason to hand out 0, 1 and 2 in any particular order
func _same_partition(a, b):
	if a.size() != b.size():
		return false
	var forward = {}
	var backward = {}
	for i in a.size():
		if forward.has(a[i]) and forward[a[i]] != b[i]:
			return false
		if backward.has(b[i]) and backward[b[i]] != a[i]:
			return false
		forward[a[i]] = b[i]
		backward[b[i]] = a[i]
	return true

# write a handmade file and hand it to a model, for the guards on the file itself
func _load_written(content, model = null):
	var path = "user://dtda_ml_test_kmeans_handmade.json"
	var file = FileAccess.open(path, FileAccess.WRITE)
	file.store_string(content)
	file.close()
	if model == null:
		model = DTDAKMeans.new()
	return model._load(path)

func _run(t):
	var data = _blobs()
	var X = data[0]
	var planted = data[1]

	t.section("K-Means, learning without labels")
	var km = DTDAKMeans.new(3)
	km._set_seed(1)
	km._fit(X)
	var groups = km._predict(X)
	t.check_equal("one group for every row", groups.size(), X.size())
	var out_of_range = 0
	var seen = {}
	for group in groups:
		seen[group] = true
		if group < 0 or group >= 3:
			out_of_range += 1
	t.check_equal("every answer is one of the k groups", out_of_range, 0)
	t.check_equal("and all k of them are used", seen.size(), 3)
	var twin = DTDAKMeans.new(3)
	twin._set_seed(1)
	t.check_equal("_fit_predict is _fit then _predict", twin._fit_predict(X), groups)

	t.section("K-Means, finding a structure that was planted")
	# The point of the model, and it is exact rather than statistical: three blobs far
	# apart are recovered on every one of forty seeds, so this counts the misses and
	# expects none. A single seed would prove nothing about the next one
	var missed = 0
	for s in 40:
		var trial = DTDAKMeans.new(3)
		trial._set_seed(s + 1)
		if not _same_partition(trial._fit_predict(X), planted):
			missed += 1
	t.check_equal("forty seeds, forty times the planted grouping", missed, 0)
	# and the centres come back where they were planted, in the unit of the data
	var centres = km._get_centroids()
	t.check_equal("one centre per group", centres.size(), 3)
	var unplaced = 0
	for planted_centre in BLOB_CENTRES:
		var nearest = 99.0
		for centre in centres:
			var distance = sqrt((centre[0] - planted_centre[0]) ** 2 + (centre[1] - planted_centre[1]) ** 2)
			if distance < nearest:
				nearest = distance
		# the blobs are ten apart and one wide, so one unit is a wide berth
		if nearest > 1.0:
			unplaced += 1
	t.check_equal("every planted centre has a learned centre on it", unplaced, 0)

	# What the iteration is for. A converged fit has every centre sitting exactly on
	# the middle of the rows it holds; a fit that never moved its centres would leave
	# them on the rows the start happened to pick, and could still hand out the right
	# grouping by luck. Standardising is affine, so the middle can be taken in the
	# unit of the data and compared against what _get_centroids() answers
	var learned = km._get_centroids()
	var off_centre = 0
	for c in learned.size():
		var totals = [0.0, 0.0]
		var count = 0
		for i in X.size():
			if groups[i] == c:
				totals[0] += X[i][0]
				totals[1] += X[i][1]
				count += 1
		if count == 0:
			continue
		# 1e-9 is far tighter than any move the iteration would still have to make on
		# columns counted in units, and loose enough not to rest on the last bit
		# asked the way round that counts a centre only when it is provably on the
		# middle: a nan answers false to every comparison, so disqualifying by "too
		# far" would read a centre that is not a number at all as landing exactly right
		var first_on = abs(learned[c][0] - totals[0] / float(count)) <= 1e-9
		var second_on = abs(learned[c][1] - totals[1] / float(count)) <= 1e-9
		if not (first_on and second_on):
			off_centre += 1
	t.check_equal("every centre sits on the middle of what it holds", off_centre, 0)

	t.section("K-Means, the unit a column is written in")
	# distances are euclidean, so without the scaler inside, a column multiplied by a
	# thousand would drown the column the groups actually live in
	var small = DTDAKMeans.new(3)
	small._set_seed(4)
	var small_groups = small._fit_predict(_mixed(1.0))
	var big = DTDAKMeans.new(3)
	big._set_seed(4)
	var big_groups = big._fit_predict(_mixed(1000.0))
	t.check_equal("a column multiplied by a thousand gives the same answer", big_groups, small_groups)

	t.section("K-Means, inertia")
	# no tolerance: _fit() leaves behind the very number _inertia_of() computes
	t.check_near("_fit leaves the inertia of the rows it was given", km._inertia_of(X), km.inertia, 0.0)
	# one group has to hold everything, three groups hold three blobs: the gap is the
	# whole spread of the data against nothing at all
	var lump = DTDAKMeans.new(1)
	lump._set_seed(1)
	lump._fit(X)
	t.check("a tighter grouping has a lower inertia", km.inertia < lump.inertia)
	# as many groups as there are rows: every row is its own centre and sits on it
	var each = DTDAKMeans.new(4)
	each._set_seed(1)
	each._fit([[0.0, 0.0], [1.0, 5.0], [9.0, 2.0], [4.0, 7.0]])
	t.check_near("a group per row leaves nothing to measure", each.inertia, 0.0, 1e-9)

	t.section("K-Means, running several times over")
	# Run one of several starts from the very same generator state as a lone run, so
	# more runs can never do worse. That one is exact and holds seed by seed
	var line = _line()
	var worse = 0
	var better = 0
	for s in 12:
		var one = DTDAKMeans.new(4, 100, 1)
		one._set_seed(s + 1)
		one._fit(line)
		var many = DTDAKMeans.new(4, 100, 8)
		many._set_seed(s + 1)
		many._fit(line)
		# the same way round, and for the same reason: an inertia that is not provably
		# no worse counts as worse, a nan included
		if not (many.inertia <= one.inertia + 1e-9):
			worse += 1
		if many.inertia < one.inertia - 1e-9:
			better += 1
	t.check_equal("more runs are never worse than one", worse, 0)
	# and they are not idle either. Measured over sixty seeds on this data, eight runs
	# beat one on fifty five of them; the worst group of twelve seeds was ten, so six
	# is far below anything seen and far above nothing
	t.check("and on this data they are usually better", better >= 6)

	t.section("K-Means, where the centres start")
	# k-means++ weighs a row by its distance to the nearest centre already chosen, so
	# a row that is already a centre weighs nothing and the draw does not return it.
	# These twenty four rows are all distinct, so the draw is all there is to it, the
	# fallback for data with fewer distinct rows than k being tested further down.
	# One draw of three out of twenty four would repeat by chance too rarely to
	# notice, two hundred draws leave a careless draw no way through
	var scaler = DTDAScaler.new()
	var scaled = scaler._fit_transform(X)
	var starter = DTDAKMeans.new(3)
	starter._set_seed(9)
	starter.n = 2
	var repeated = 0
	var miscounted = 0
	for i in 200:
		var start = starter._initial_centroids(scaled)
		if start.size() != 3:
			miscounted += 1
		for a in start.size():
			for b in range(a + 1, start.size()):
				if start[a] == start[b]:
					repeated += 1
	t.check_equal("no draw among two hundred takes the same row twice", repeated, 0)
	t.check_equal("and every one of them draws k centres", miscounted, 0)

	t.section("K-Means, determinism")
	var seeded_a = DTDAKMeans.new(4, 100, 3)
	seeded_a._set_seed(77)
	seeded_a._fit(line)
	var seeded_b = DTDAKMeans.new(4, 100, 3)
	seeded_b._set_seed(77)
	seeded_b._fit(line)
	t.check_equal("the same seed finds the same groups", seeded_b._predict(line), seeded_a._predict(line))
	t.check_near("and the same inertia", seeded_b.inertia, seeded_a.inertia, 0.0)
	var before_reset = seeded_a._predict(line)
	seeded_a._reset()
	t.check_empty("_reset forgets the centres", seeded_a._predict(line))
	seeded_a._fit(line)
	t.check_equal("K-Means _reset replays the same draws", seeded_a._predict(line), before_reset)

	t.section("K-Means, edges")
	var one_group = DTDAKMeans.new(1)
	one_group._set_seed(1)
	var lumped = one_group._fit_predict(X)
	var not_zero = 0
	for group in lumped:
		if group != 0:
			not_zero += 1
	t.check_equal("a single group holds everything", not_zero, 0)
	# one pass of Lloyd is a poor fit and still has to be a usable one
	var hurried = DTDAKMeans.new(3, 1)
	hurried._set_seed(1)
	t.check_equal("one iteration still answers a group per row", hurried._fit_predict(X).size(), X.size())
	# a column that never changes would be divided by zero without the scaler's guard
	var flat = DTDAKMeans.new(2)
	flat._set_seed(1)
	var flat_groups = flat._fit_predict([[7.0, 0.0], [7.0, 1.0], [7.0, 8.0], [7.0, 9.0]])
	t.check_equal("a column that never changes does not break the fit", flat_groups.size(), 4)
	t.check("and the groups still follow the column that does",
		flat_groups[0] == flat_groups[1] and flat_groups[2] == flat_groups[3] and flat_groups[0] != flat_groups[2])

	# rows that are all the same leave the start with nothing to spread out over, and
	# the fit still owes k centres rather than the one it could get away with
	var same_rows = DTDAKMeans.new(3)
	same_rows._set_seed(1)
	var same_groups = same_rows._fit_predict([[5.0, 5.0], [5.0, 5.0], [5.0, 5.0], [5.0, 5.0]])
	t.check_equal("rows that are all the same still get k centres", same_rows._get_centroids().size(), 3)
	t.check_near("and leave nothing to measure", same_rows.inertia, 0.0, 1e-9)
	t.check_equal("with every row in the same group", same_groups, [0, 0, 0, 0])

	# Two values and three groups. k-means++ runs out of distance to spread the starts
	# over, the fallback takes a row that is already a centre, two centres land on the
	# same spot and the third group is left holding nothing. A group holding nothing
	# keeps the centre it had rather than being moved anywhere, so every centre stays
	# on one of the two values the data holds and none of them drifts to the middle.
	# Twenty seeds rather than one, the first draw deciding which value comes first
	var adrift = 0
	for s in 20:
		var starved = DTDAKMeans.new(3)
		starved._set_seed(s + 1)
		starved._fit([[0.0], [0.0], [10.0], [10.0]])
		for centre in starved._get_centroids():
			# asked the way round that counts a centre only when it is provably on
			# one of the two values: every comparison against a nan answers false,
			# so the other way round would let a nan through as if it were fine
			var on_a_value = abs(centre[0]) <= 1e-9 or abs(centre[0] - 10.0) <= 1e-9
			if not on_a_value:
				adrift += 1
	t.check_equal("a group left holding nothing keeps the centre it had", adrift, 0)

	t.section("K-Means, saving and loading")
	var path = "user://dtda_ml_test_kmeans.json"
	t.check("K-Means _save reports a success", km._save(path))
	var back = DTDAKMeans.new()
	t.check("K-Means _load reports a success", back._load(path))
	t.check_equal("a reloaded model answers the same groups", back._predict(X), km._predict(X))
	# 1e-9 is far tighter than anything the model could get wrong on a centre whose
	# columns are counted in units, and loose enough not to rest on how the engine
	# happens to write a float down
	t.check_near_array("and holds the same centres, in the unit of the data",
		back._get_centroids()[0], km._get_centroids()[0], 1e-9)
	t.check_near("the inertia comes back", back.inertia, km.inertia, 1e-9)
	t.check_equal("and so do the settings",
		[back.k, back.max_iterations, back.num_runs], [km.k, km.max_iterations, km.num_runs])

	t.section("K-Means guards (the errors below are expected)")
	t.check_empty("K-Means _predict before _fit", DTDAKMeans.new()._predict([[1.0, 2.0]]))
	t.check_equal("K-Means _save before _fit fails", DTDAKMeans.new()._save(path), false)
	t.check_empty("_get_centroids before _fit", DTDAKMeans.new()._get_centroids())
	t.check_near("_inertia_of before _fit", DTDAKMeans.new()._inertia_of([[1.0, 2.0]]), 0.0)
	var no_data = DTDAKMeans.new()
	no_data._fit([])
	t.check_empty("K-Means _fit with no data leaves it unfitted", no_data._predict([[1.0]]))
	var no_group = DTDAKMeans.new(0)
	no_group._fit(X)
	t.check_empty("_fit for no group at all leaves it unfitted", no_group._predict([[1.0, 2.0]]))
	var too_few = DTDAKMeans.new(5)
	too_few._fit([[0.0, 0.0], [1.0, 1.0]])
	t.check_empty("_fit with fewer rows than groups leaves it unfitted", too_few._predict([[1.0, 2.0]]))
	var no_run = DTDAKMeans.new(2, 100, 0)
	no_run._fit(X)
	t.check_empty("_fit for no run at all leaves it unfitted", no_run._predict([[1.0, 2.0]]))
	# and a fit that is refused leaves a model that was working exactly as it was,
	# the way a refused file does. Four different refusals answer for this one
	var standing = DTDAKMeans.new(3)
	standing._set_seed(1)
	standing._fit(X)
	var standing_before = standing._predict(X)
	standing.k = 0
	standing._fit(X)
	standing.k = 99
	standing._fit(X)
	standing.k = 3
	standing.num_runs = 0
	standing._fit(X)
	standing.num_runs = 5
	standing._fit([])
	t.check_equal("a refused fit leaves the model answering as before",
		standing._predict(X), standing_before)

	# a file written by another model, saved here so this suite stays self contained
	var other = DTDAKNN.new(1)
	other._fit([[0]], [1])
	var other_path = "user://dtda_ml_test_not_a_kmeans.json"
	other._save(other_path)
	t.check_equal("K-Means _load refuses another kind of model", km._load(other_path), false)
	t.check_equal("K-Means _load refuses a missing file",
		km._load("user://no_such_kmeans.json"), false)
	# a file a model could read from end to end, wrong on the "model" field alone: the
	# KNN file above is turned away by the guards on the structure long before the
	# name is ever weighed
	t.check_equal("DTDAKMeans refuses a file that only lies about its model name",
		_load_written(SOUND.replace("DTDAKMeans", "NotAKMeans"), km), false)
	t.check_equal("K-Means _load refuses another format version",
		_load_written(SOUND.replace('"version": 1', '"version": 99'), km), false)
	# check_equal against false, not "not <call>": a call that raises answers null,
	# and "not null" is true, which would turn a crash into a pass
	t.check_equal("_load refuses centres that are not a list",
		_load_written(SOUND.replace(CENTRES_FIELD, '"centroids": 5'), km), false)
	t.check_equal("_load refuses an empty list of centres",
		_load_written(SOUND.replace(CENTRES_FIELD, '"centroids": []'), km), false)
	t.check_equal("_load refuses a centre that is not a row of numbers",
		_load_written(SOUND.replace(CENTRES_FIELD, '"centroids": [[0.0, 1.0], [0.0, "nope"]]'), km), false)
	t.check_equal("_load refuses centres of different widths",
		_load_written(SOUND.replace(CENTRES_FIELD, '"centroids": [[0.0, 1.0], [0.0]]'), km), false)
	t.check_equal("_load refuses an inertia that is not a number",
		_load_written(SOUND.replace('"inertia": 1.5', '"inertia": "nope"'), km), false)
	t.check_equal("_load refuses a scaler it cannot use",
		_load_written(SOUND.replace(SCALER_FIELD, '"scaler": {"mode": 0, "offsets": [0.0, 0.0], "scales": [1.0, 0.0]}'), km), false)
	# _predict() scales a row and then measures it against the centres: a scaler of
	# two columns and centres of three would read past the end of one of them
	t.check_equal("_load refuses a scaler as wide as the centres are not",
		_load_written(SOUND.replace(CENTRES_FIELD, '"centroids": [[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]]'), km), false)
	# and none of those refusals may take the standing model down with it: every one
	# of them above was handed to km, the model fitted at the top of this suite, so
	# the two lines below are answering for eleven different refusals and not for one
	t.check_equal("a refused file leaves the model answering as before", km._predict(X), groups)
	t.check_near("and leaves its inertia alone", km._inertia_of(X), km.inertia, 0.0)
