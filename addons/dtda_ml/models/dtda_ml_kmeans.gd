extends DTDATools

class_name DTDAKMeans

# === K-Means === #
# The first model here that learns without labels. fit() is handed rows and nothing
# else, and works out which of k groups each row belongs to.
#
# Distances are euclidean, so a column counted in tens of thousands would drown a
# column counted in units. The rows are standardised internally by a DTDAScaler, the
# way DTDALinReg does it: nothing has to be scaled beforehand, and the unit a column
# is written in does not change the answer. get_centroids() hands the centres back
# in the unit of the training data, which is what a game wants to draw.
#
# Where the centres start decides where they end, and a poor start stays poor: it is
# not noise that averages out over the iterations. Two well worn answers, both here:
#  - k-means++, which draws the first centre among the rows and each of the others
#    with a weight of its squared distance to the nearest centre already chosen, so
#    the starts spread out instead of huddling
#  - several runs from several starts, keeping the one with the lowest inertia
# Both draw on one generator, so set_seed() replays a whole fit.

const FORMAT_VERSION = 1

var k
var max_iterations
var num_runs
var m
var n
# the centres, in the scaled space the model works in, null until fit()
var centroids
# the sum of the squared distances from every training row to its centre. Without
# labels to compare against it is the only measure of quality there is
var inertia
var scaler
# its own generator, so a run can be replayed with set_seed()
var rng
# the seed given to set_seed(), replayed by reset(), null when none was asked for
var start_seed

func _init(kmeans_k := 3, kmeans_max_iterations := 100, kmeans_num_runs := 5):
	k = kmeans_k
	max_iterations = kmeans_max_iterations
	num_runs = kmeans_num_runs
	rng = RandomNumberGenerator.new()
	start_seed = null

# fix the draws, for a reproducible fit
# reset() puts the generator back on that same seed
func set_seed(value):
	start_seed = value
	rng.seed = value

# forget the centres and put the generator back where it started
func reset():
	centroids = null
	inertia = null
	if start_seed != null:
		rng.seed = start_seed

# squared, because the square root would change neither which centre is nearest nor
# the order of two distances, and inertia is defined on the squares anyway
func _square_distance(row, centre):
	var total = 0.0
	for i in centre.size():
		total += (row[i] - centre[i]) ** 2
	return total

# which centre a scaled row belongs to, and how far it sits from it
func _nearest(row, centres):
	var best = 0
	var best_distance = INF
	for i in centres.size():
		var distance = _square_distance(row, centres[i])
		# strict, so a tie keeps the lower index and the same row always answers
		# the same group
		if distance < best_distance:
			best = i
			best_distance = distance
	return [best, best_distance]

# k-means++ : the first centre is a row taken at random, then each new centre is drawn
# among the rows with a weight of its squared distance to the nearest centre already
# chosen. A row that is already a centre weighs zero, so the draw never returns it a
# second time, and rows far from everything chosen so far come up often. The fallback
# further down is the one exception, and it is not a draw
func _initial_centroids(rows):
	var centres = [rows[rng.randi() % rows.size()].duplicate()]
	while centres.size() < k:
		var weights = []
		var total = 0.0
		for row in rows:
			var weight = _nearest(row, centres)[1]
			weights.push_back(weight)
			total += weight
		# every row sits exactly on a centre already, which happens when the data
		# holds fewer distinct rows than k. There is nothing left to spread out, so
		# the remaining centres are taken in order rather than drawn from nothing
		if total == 0.0:
			for row in rows:
				if centres.size() >= k:
					break
				centres.push_back(row.duplicate())
			break
		var target = rng.randf() * total
		var running = 0.0
		# the last row is the fallback: floating point can leave the running sum a
		# hair under the target on the very last step
		var chosen = rows.size() - 1
		for i in rows.size():
			running += weights[i]
			if running >= target:
				chosen = i
				break
		centres.push_back(rows[chosen].duplicate())
	return centres

# one run of Lloyd: put every row with its nearest centre, move every centre to the
# middle of what it holds, and stop when nobody changed group
func _one_run(rows):
	var centres = _initial_centroids(rows)
	var labels = []
	for i in rows.size():
		labels.push_back(-1)
	for step in max_iterations:
		var moved = false
		for i in rows.size():
			var nearest = _nearest(rows[i], centres)[0]
			if nearest != labels[i]:
				labels[i] = nearest
				moved = true
		# nobody changed group, so no centre would move either
		if not moved:
			break
		for c in centres.size():
			var totals = _array_zeros(n)
			var count = 0
			for i in rows.size():
				if labels[i] == c:
					for u in n:
						totals[u] += rows[i][u]
					count += 1
			# an empty group has no middle to move to: its centre stays where it is,
			# the gentlest of the usual answers, and it never invents a row
			if count == 0:
				continue
			for u in n:
				centres[c][u] = totals[u] / float(count)
	return centres

func _total_inertia(rows, centres):
	var total = 0.0
	for row in rows:
		total += _nearest(row, centres)[1]
	return total

func fit(newX):
	# The rows are weighed before a single field is written: a fit that took them as
	# they came would leave a working model holding a nan, or half rewritten by a
	# raise in the middle. Answers false when it refuses, true when it fitted
	if not _check_matrix(newX, "DTDAKMeans"):
		return false
	if k <= 0:
		push_error("DTDAKMeans: fit() called for %d groups" % k)
		return false
	if num_runs <= 0:
		push_error("DTDAKMeans: fit() called for %d runs" % num_runs)
		return false
	if newX.size() < k:
		push_error("DTDAKMeans: fit() got %d rows for %d groups" % [newX.size(), k])
		return false
	m = newX.size()
	n = newX[0].size()
	# built aside, so a fit that never reaches the end leaves the standing model alone
	var fitted_scaler = DTDAScaler.new()
	var rows = fitted_scaler.fit_transform(newX)
	var best = null
	var best_inertia = INF
	for run in num_runs:
		var centres = _one_run(rows)
		var run_inertia = _total_inertia(rows, centres)
		# strict, so the first of two equally good runs is the one kept
		if run_inertia < best_inertia:
			best = centres
			best_inertia = run_inertia
	scaler = fitted_scaler
	centroids = best
	inertia = best_inertia
	return true

func predict(newX):
	if not _check_fitted("DTDAKMeans", centroids):
		return []
	var pred = []
	for row in scaler.transform(newX):
		pred.push_back(_nearest(row, centroids)[0])
	return pred

func fit_predict(newX):
	fit(newX)
	return predict(newX)

# the inertia of any set of rows against the centres already learned. Lower is
# tighter, and it only ever compares groupings of the same rows: it falls as k rises
# whatever the grouping is worth, so it cannot be read as a score on its own
func inertia_of(newX):
	if not _check_fitted("DTDAKMeans", centroids, "inertia_of()"):
		return 0.0
	return _total_inertia(scaler.transform(newX), centroids)

# the centres in the unit of the training data, rather than the scaled space the
# model works in
func get_centroids():
	if not _check_fitted("DTDAKMeans", centroids, "get_centroids()"):
		return []
	return scaler.inverse_transform(centroids)

func to_dict():
	if not _check_fitted("DTDAKMeans", centroids, "save()"):
		return {}
	return {
		"model": "DTDAKMeans",
		"version": FORMAT_VERSION,
		"k": k,
		"max_iterations": max_iterations,
		"num_runs": num_runs,
		"inertia": inertia,
		"centroids": centroids,
		"scaler": scaler.to_dict(),
	}

func from_dict(data):
	if not _check_model_name(data, "DTDAKMeans"):
		return false
	# int() because a version read back from JSON carries as a float
	var version = int(data.get("version", 0))
	if version != FORMAT_VERSION:
		push_error("DTDAKMeans: this file is written in format %d, this model reads format %d" % [version, FORMAT_VERSION])
		return false
	# Everything is read aside first and only takes the place of the standing model
	# once the whole file is known to be readable, and what is read is what a
	# prediction computes with: every centre, and the scaler that brings a row into
	# the space those centres live in
	var saved_centroids = data.get("centroids")
	if typeof(saved_centroids) != TYPE_ARRAY or saved_centroids.size() == 0:
		push_error("DTDAKMeans: the saved model has no centres")
		return false
	for centre in saved_centroids:
		if not _check_number_array(centre, "DTDAKMeans", "centre"):
			return false
		if centre.size() != saved_centroids[0].size():
			push_error("DTDAKMeans: the saved centres do not all hold the same number of columns")
			return false
	var saved_inertia = data.get("inertia", 0)
	if not _check_number(saved_inertia, "DTDAKMeans", "inertia"):
		return false
	var saved_scaler = DTDAScaler.new()
	if not saved_scaler.from_dict(data.get("scaler", {})):
		return false
	# predict() scales a row and then measures it against the centres, so a scaler
	# of one width and centres of another would read past the end of one of them
	if saved_scaler.offsets.size() != saved_centroids[0].size():
		push_error("DTDAKMeans: the saved scaler holds %d columns and the centres %d" % [saved_scaler.offsets.size(), saved_centroids[0].size()])
		return false
	k = int(data.get("k", saved_centroids.size()))
	max_iterations = int(data.get("max_iterations", max_iterations))
	num_runs = int(data.get("num_runs", num_runs))
	inertia = saved_inertia
	centroids = saved_centroids
	n = saved_centroids[0].size()
	scaler = saved_scaler
	return true


# === The older names === #
# Every method above used to carry a leading underscore, which in Godot marks a
# method as virtual or private: the engine calls _ready() and _process(), you do not.
# The names below are the ones that shipped, kept working so nothing that already
# calls them breaks. They only forward. Prefer the ones without the underscore.

func _set_seed(value):
	return set_seed(value)

func _reset():
	return reset()

func _fit(newX):
	return fit(newX)

func _predict(newX):
	return predict(newX)

func _fit_predict(newX):
	return fit_predict(newX)

func _inertia_of(newX):
	return inertia_of(newX)

func _get_centroids():
	return get_centroids()



# === End K-Means === #
