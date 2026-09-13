extends DTDATools

class_name DTDAKMeans

# The first model here that learns without labels: fit() is handed rows and nothing else, and
# works out which of k groups each row belongs to. Distances are euclidean, so a column in
# tens of thousands would drown one in units: the rows are standardised internally by a
# DTDAScaler, as DTDALinReg does, and get_centroids() hands the centres back in the unit of
# the training data, which is what a game wants to draw. Where the centres start decides where they end, and a poor start stays poor rather than averaging out over the iterations,
# hence k-means++ and several runs kept on the lowest inertia. Both draw on one generator, so set_seed() replays a whole fit.

const FORMAT_VERSION = 1

var k: int
var max_iterations: int
var num_runs: int
var m: int = 0
var n: int = 0
var centroids
# the sum of the squared distances from every training row to its centre. Without labels to compare against it is the only measure of quality there is
var inertia
var scaler
var rng: RandomNumberGenerator
var start_seed

func _init(kmeans_k: int = 3, kmeans_max_iterations: int = 100, kmeans_num_runs: int = 5) -> void:
	k = kmeans_k
	max_iterations = kmeans_max_iterations
	num_runs = kmeans_num_runs
	rng = RandomNumberGenerator.new()
	start_seed = null

# fix the draws for a reproducible fit; reset() replays this same seed
func set_seed(value: int) -> void:
	start_seed = value
	rng.seed = value

func reset() -> void:
	centroids = null
	inertia = null
	if start_seed != null:
		rng.seed = start_seed

# squared, because the square root would change neither which centre is nearest nor the order of two distances, and inertia is defined on the squares anyway
func _square_distance(row, centre) -> float:
	var total: float = 0.0
	for i in centre.size():
		total += (row[i] - centre[i]) ** 2
	return total

func _nearest(row, centres) -> Array:
	var best: int = 0
	var best_distance: float = INF
	for i in centres.size():
		var distance: float = _square_distance(row, centres[i])
		# strict, so a tie keeps the lower index and the same row always answers the same group
		if distance < best_distance:
			best = i
			best_distance = distance
	return [best, best_distance]

# k-means++ : the first centre is a row taken at random, then each new centre is drawn among
# the rows with a weight of its squared distance to the nearest centre already chosen. A row
# already a centre weighs zero, so the draw never returns it twice, and rows far from everything chosen come up often. The fallback further down is the one exception, and it is not a draw
func _initial_centroids(rows) -> Array:
	var centres = [rows[rng.randi() % rows.size()].duplicate()]
	while centres.size() < k:
		var weights: Array = []
		var total: float = 0.0
		for row in rows:
			var weight: float = _nearest(row, centres)[1]
			weights.push_back(weight)
			total += weight
			# every row sits exactly on a centre already, which happens when the data holds fewer distinct rows than k: nothing left to spread out, so the remaining centres are taken in order rather than drawn from nothing
		if total == 0.0:
			for row in rows:
				if centres.size() >= k:
					break
				centres.push_back(row.duplicate())
			break
		var target: float = rng.randf() * total
		var running: float = 0.0
		# the last row is the fallback: floating point can leave the running sum a hair under the target on the very last step
		var chosen = rows.size() - 1
		for i in rows.size():
			running += weights[i]
			if running >= target:
				chosen = i
				break
		centres.push_back(rows[chosen].duplicate())
	return centres

# one run of Lloyd: put every row with its nearest centre, move every centre to the middle of what it holds, and stop when nobody changed group
func _one_run(rows) -> Array:
	var centres = _initial_centroids(rows)
	var labels: Array = []
	for i in rows.size():
		labels.push_back(-1)
	for step in max_iterations:
		var moved: bool = false
		for i in rows.size():
			var nearest = _nearest(rows[i], centres)[0]
			if nearest != labels[i]:
				labels[i] = nearest
				moved = true
		if not moved:
			break
		for c in centres.size():
			var totals = _array_zeros(n)
			var count: int = 0
			for i in rows.size():
				if labels[i] == c:
					for u in n:
						totals[u] += rows[i][u]
					count += 1
			# an empty group has no middle to move to: its centre stays where it is, the gentlest of the usual answers, and it never invents a row
			if count == 0:
				continue
			for u in n:
				centres[c][u] = totals[u] / float(count)
	return centres

func _packed_rows(rows) -> Array:
	var packed: Array = []
	for row in rows:
		packed.push_back(PackedFloat64Array(row))
	return packed

func _plain_rows(rows) -> Array:
	var plain: Array = []
	for row in rows:
		plain.push_back(Array(row))
	return plain

func _total_inertia(rows, centres) -> float:
	var total: float = 0.0
	for row in rows:
		total += _nearest(row, centres)[1]
	return total

func fit_begin(newX) -> bool:
	# the rows are weighed before a single field is written: a fit that took them as they came would leave a working model holding a nan, or half rewritten by a raise in the middle
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
	# the slice is one run from one set of starts, built aside, so a training that never reaches the end leaves the standing model exactly as it was
	var fitted_scaler := DTDAScaler.new()
	_fit_work = {
		"rows": _packed_rows(fitted_scaler.fit_transform(newX)),
		"scaler": fitted_scaler,
		"best": null,
		"best_inertia": INF,
		"run": 0,
	}
	return true

func _model_name() -> String:
	return "DTDAKMeans"

func fit_step() -> float:
	if _fit_work == null:
		push_error("DTDAKMeans: fit_step() called with no training under way")
		return 1.0
	var work: Dictionary = _fit_work
	if int(work["run"]) < num_runs:
		var centres = _one_run(work["rows"])
		var run_inertia: float = _total_inertia(work["rows"], centres)
		# strict, so the first of two equally good runs is the one kept
		if run_inertia < float(work["best_inertia"]):
			work["best"] = centres
			work["best_inertia"] = run_inertia
		work["run"] = int(work["run"]) + 1
	if int(work["run"]) < num_runs:
		return float(work["run"]) / float(num_runs)
	scaler = work["scaler"]
	centroids = _plain_rows(work["best"])
	inertia = work["best_inertia"]
	_fit_work = null
	return 1.0

func fit(newX) -> bool:
	if not fit_begin(newX):
		return false
	return _fit_every_step()

func predict(newX) -> Array:
	if not _check_fitted("DTDAKMeans", centroids):
		return []
	var pred: Array = []
	for row in scaler.transform(newX):
		pred.push_back(_nearest(row, centroids)[0])
	return pred

func fit_predict(newX) -> Array:
	fit(newX)
	return predict(newX)

# the inertia of any set of rows against the centres already learned. Lower is tighter, and it only ever compares groupings of the same rows: it falls as k rises whatever the grouping is worth, so it cannot be read as a score on its own
func inertia_of(newX) -> float:
	if not _check_fitted("DTDAKMeans", centroids, "inertia_of()"):
		return 0.0
	return _total_inertia(scaler.transform(newX), centroids)

# the centres in the unit of the training data, rather than the scaled space the model works in
func get_centroids() -> Array:
	if not _check_fitted("DTDAKMeans", centroids, "get_centroids()"):
		return []
	return scaler.inverse_transform(centroids)

func to_dict() -> Dictionary:
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

func from_dict(data) -> bool:
	if not _check_model_name(data, "DTDAKMeans"):
		return false
	# int() because a version read back from JSON carries as a float
	var version = int(data.get("version", 0))
	if version != FORMAT_VERSION:
		push_error("DTDAKMeans: this file is written in format %d, this model reads format %d" % [version, FORMAT_VERSION])
		return false
	# everything is read aside and only takes the place of the standing model once the whole file is known to be readable: every centre, and the scaler that brings a row into the space those centres live in
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
	# predict() scales a row and then measures it against the centres, so a scaler of one width and centres of another would read past the end of one of them
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


# the older underscored spellings, kept working for what already calls them; they only forward

func _set_seed(value):
	set_seed(value)

func _reset():
	reset()

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

