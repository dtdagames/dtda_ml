
const PLAN = 38

const DATA_KNN = [[2, 4, 2, 1, 0, 0, 3], [2, 2, 4, 0, 0, 0, 4], [4, 2, 1, 1, 0, 1, 5], [2, 2, 4, 0, 1, 1, 6]]
const DATA_LINR = [[1.6, 40000], [4.6, 60000], [4.2, 58000], [4.1, 59000], [5.4, 80000],
	[8.1, 100000], [8.9, 110000], [9.2, 110000], [9.3, 114000], [10.2, 121000]]
const DATA_LOGR = [[2, 4, 2, 1, 0, 0, 0], [2, 2, 4, 0, 0, 0, 0], [4, 2, 1, 1, 0, 1, 1], [2, 2, 4, 0, 1, 1, 1]]
const CLASS_TEST = [[1, 3, 1, 0, 1, 0], [2, 2, 4, 1, 1, 1], [4, 1, 1, 0, 1, 0]]
const PROBE_LIN = [[7.2], [9.0], [11.1]]
const NAN_ROWS = [[1.0, 2.0], [2.0, 1.0], [8.0, NAN]]
const SOUND_ROWS = [[1.0, 2.0], [2.0, 1.0], [8.0, 9.0]]
const UNIT = '{"mode": 0, "offsets": [0.0], "scales": [1.0]}'
const ZERO = '{"mode": 0, "offsets": [0.0], "scales": [0.0]}'
const LIN_OK = '"x_scaler": ' + UNIT + ', "y_scaler": ' + UNIT
const ONE_OK = '"scaler": ' + UNIT
const ONE_ZERO = '"scaler": ' + ZERO
# the four whose refusal is read through the invariant rather than one by one: each ruins the model outright when its guard goes
const KNN_BAD = ['"num_neighbors": 0, "X": [[0]], "Y": [3]', '"num_neighbors": 1, "X": [], "Y": []',
	'"num_neighbors": 1, "X": [[0], [1]], "Y": [3]', '"num_neighbors": 1, "X": [[0], ["nope"]], "Y": [3, 4]']

func _scale_rows(rows, factor):
	return rows.map(func(row): return row.map(func(value): return value * factor))

func _fitted(model, X, y):
	model.fit(X, y)
	return model

# a file readable from end to end and wrong in exactly one field: two wrong at once and either guard could be the one answering
func _file(name, fields):
	return '{"model": "%s", "version": 1, %s}' % [name, fields]

func _load_written(content, model):
	var file = FileAccess.open("user://dtda_ml_test_handmade.json", FileAccess.WRITE)
	file.store_string(content)
	file.close()
	return model.load("user://dtda_ml_test_handmade.json")

func _refused(t, name, model, content):
	t.check_equal(name, _load_written(content, model), false)

func _unmoved(t, name, model, probe, action):
	var before = model.predict(probe)
	action.call()
	t.check_near_array(name, model.predict(probe), before, 0.001)

func _bad_fits(model, weighs_labels):
	model.fit(NAN_ROWS, [0, 1, 1])
	model.fit(SOUND_ROWS, [0])
	if weighs_labels:
		model.fit([[1.0], [2.0]], ["red", "blue"])

func _knn_bad_files(t, model):
	_refused(t, "KNN refuses training rows that are not a list", model, _file("DTDAKNN", '"num_neighbors": 1, "X": "nope", "Y": [3]'))
	_refused(t, "KNN refuses labels that are not a list", model, _file("DTDAKNN", '"num_neighbors": 1, "X": [[0]], "Y": "nope"'))
	_refused(t, "KNN refuses a neighbour count that is not a number", model, _file("DTDAKNN", '"num_neighbors": "nope", "X": [[0]], "Y": [3]'))
	for fields in KNN_BAD:
		_load_written(_file("DTDAKNN", fields), model)

func _run(t):
	var ml = DTDATools.new()

	t.section("KNN")
	var knn = _fitted(DTDAKNN.new(3), ml.drop_variable(DATA_KNN, 6), ml.get_variable(DATA_KNN, 6))
	t.check_near_array("predicts the expected labels", knn.predict([[1, 4, 1, 1, 0, 0], [2, 2, 4, 1, 1, 1], [4, 1, 1, 0, 1, 0]]), [3, 6, 5])
	# the two nearest neighbours out of three carry label 1, the closest carries 0: a 1-NN would answer 0, the majority vote answers 1
	t.check_near_array("takes the majority, not the closest neighbour", _fitted(DTDAKNN.new(3), [[0.1], [0.5], [0.6]], [0, 1, 1]).predict([[0.0]]), [1])
	t.check_near_array("the distance uses every feature", _fitted(DTDAKNN.new(1), [[0, 0], [0, 5]], [10, 20]).predict([[0, 5]]), [20])
	# the label rather than the size: reading past the end of the neighbour list answers a list holding one null, whose size is one too
	t.check_equal("k larger than the training set", _fitted(DTDAKNN.new(10), [[0], [1]], [7, 8]).predict([[0]]), [7])
	# a KNN only hands a label back, so a label naming a class is not a fault: without this, tightening the three fits that do weigh their labels into all four would go through without an objection
	var named = _fitted(DTDAKNN.new(1), [[0.0, 0.0], [0.5, 0.5], [9.0, 9.0], [9.5, 9.5]], ["cave", "cave", "camp", "camp"])
	t.check_equal("KNN takes labels that name a class and hands the name back", named.predict([[0.2, 0.2], [9.2, 9.2]]), ["cave", "camp"])

	t.section("Linear and logistic regression, SVM")
	var X_lin = ml.drop_variable(DATA_LINR, 1)
	var y_lin = ml.get_variable(DATA_LINR, 1)
	var linreg = _fitted(DTDALinReg.new(0.01, 1000), X_lin, y_lin)
	t.check("fits the training set closely", ml.r2_score(linreg.predict(X_lin), y_lin) > 0.98)
	# standardization makes the model independent of the scale of its features, and no mutation of a single guard can tell: these three lines are what stands between the descent and a scaler regressed to an integer division
	var big = _fitted(DTDALinReg.new(0.01, 1000), _scale_rows(X_lin, 1000), y_lin)
	t.check_near_array("features x1000 give the same predictions", big.predict(_scale_rows(PROBE_LIN, 1000)), linreg.predict(PROBE_LIN), 1.0)
	var X_log = ml.drop_variable(DATA_LOGR, 6)
	var y_log = ml.get_variable(DATA_LOGR, 6)
	var logreg = _fitted(DTDALogReg.new(0.01, 1000), X_log, y_log)
	t.check_near_array("separates the training set", logreg.predict(X_log), y_log)
	# without standardization exp() overflows here
	var big_log = _fitted(DTDALogReg.new(0.01, 1000), _scale_rows(X_log, 1000), y_log)
	t.check_near_array("LogReg on features x1000 gives the same classes", big_log.predict(_scale_rows(CLASS_TEST, 1000)), [0, 1, 1])
	var svm = _fitted(DTDASVM.new(0.01, 0.01, 1000), X_log, y_log)
	t.check_near_array("predicts -1 and 1", svm.predict(CLASS_TEST), [-1, 1, 1])
	var big_svm = _fitted(DTDASVM.new(0.01, 0.01, 1000), _scale_rows(X_log, 1000), y_log)
	t.check_near_array("SVM on features x1000 gives the same classes", big_svm.predict(_scale_rows(CLASS_TEST, 1000)), [-1, 1, 1])

	t.section("Saving and loading")
	var before = linreg.predict(PROBE_LIN)
	linreg.save("user://dtda_ml_test_linreg.json")
	var reloaded = DTDALinReg.new(0.01, 1000)
	reloaded.load("user://dtda_ml_test_linreg.json")
	t.check_near_array("a reloaded model predicts the same", reloaded.predict(PROBE_LIN), before, 0.001)
	knn.save("user://dtda_ml_test_knn.json")
	# built with a different count on purpose: a receiver already holding 3 would load the same 3 whether from_dict assigns it or not, and the field would go unpinned
	var knn_back = DTDAKNN.new(1)
	knn_back.load("user://dtda_ml_test_knn.json")
	t.check_equal("a reloaded KNN takes the neighbour count from the file", knn_back.num_neighbors, 3)
	t.check_near_array("a reloaded KNN predicts the same", knn_back.predict(CLASS_TEST), knn.predict(CLASS_TEST))

	t.section("Persistence guards (the errors below are expected)")
	_refused(t, "DTDAKNN refuses a file that only lies about its model name", DTDAKNN.new(3), _file("NotAKNN", '"num_neighbors": 1, "X": [[0]], "Y": [1]'))
	_refused(t, "DTDALinReg refuses a file that only lies about its model name", DTDALinReg.new(0.01, 1000), _file("NotALinReg", '"W": [1.0], "b": 0.0, ' + LIN_OK))
	_refused(t, "DTDALogReg refuses a file that only lies about its model name", DTDALogReg.new(0.01, 1000), _file("NotALogReg", '"W": [1.0], "b": 0.0, ' + ONE_OK))
	_refused(t, "DTDASVM refuses a file that only lies about its model name", DTDASVM.new(0.01, 0.01, 1000), _file("NotASVM", '"W": [1.0], "b": 0.0, ' + ONE_OK))
	# predict() computes with every one of these numbers, so a file holding a text where the weights belong is not usable: it used to load, answering null after a cascade, and the model was ruined either way
	_refused(t, "LinReg refuses weights that are not a list", DTDALinReg.new(0.01, 1000), _file("DTDALinReg", '"W": "nope", ' + LIN_OK))
	_refused(t, "LinReg refuses an intercept that is not a number", DTDALinReg.new(0.01, 1000), _file("DTDALinReg", '"W": [1.0], "b": "nope", ' + LIN_OK))
	_refused(t, "LogReg refuses weights that are not a list", DTDALogReg.new(0.01, 1000), _file("DTDALogReg", '"W": {"a": 1}, ' + ONE_OK))
	_refused(t, "LogReg refuses an intercept that is not a number", DTDALogReg.new(0.01, 1000), _file("DTDALogReg", '"W": [1.0], "b": "nope", ' + ONE_OK))
	_refused(t, "SVM refuses weights that are not a list", DTDASVM.new(0.01, 0.01, 1000), _file("DTDASVM", '"W": 5, ' + ONE_OK))
	_refused(t, "SVM refuses an intercept that is not a number", DTDASVM.new(0.01, 0.01, 1000), _file("DTDASVM", '"W": [1.0], "b": "nope", ' + ONE_OK))

	t.section("Persistence, a refused file changes nothing (the errors below are expected)")
	# a scaler that holds a zero divides by it at the first prediction: that file used to load with a success and answer inf, and the three models below wrote the weights of a file they went on to refuse over the ones they were working with
	_unmoved(t, "and LinReg predicts what it predicted before", linreg, PROBE_LIN,
		func(): _load_written(_file("DTDALinReg", '"rate": 0.5, "iterations": 3, "W": [9.9], "b": 7.7, ' + '"x_scaler": ' + ZERO + ', "y_scaler": ' + UNIT), linreg))
	_unmoved(t, "and LogReg predicts what it predicted before", logreg, CLASS_TEST,
		func(): _load_written(_file("DTDALogReg", '"rate": 0.5, "iterations": 3, "W": [9.9], "b": 7.7, ' + ONE_ZERO), logreg))
	_unmoved(t, "and SVM predicts what it predicted before", svm, CLASS_TEST,
		func(): _load_written(_file("DTDASVM", '"lr": 0.5, "lambda": 0.5, "iter": 3, "W": [9.9], "b": 7.7, ' + ONE_ZERO), svm))
	t.check_equal("no scrap of the refused file is left behind", [linreg.rate, logreg.iterations, svm.iter], [0.01, 1000, 1000])

	t.section("Persistence, a KNN training set read out of a file (the errors below are expected)")
	# a KNN answers with the rows it kept, so those rows have to be rows of numbers, and it counts k out at every prediction rather than at fit()
	_unmoved(t, "and none of those refusals moved it", knn, CLASS_TEST, func(): _knn_bad_files(t, knn))

	t.section("A fit that is refused changes nothing (the errors below are expected)")
	# fit() is handed whatever the caller computed, and one unlucky division upstream is enough: a nan in a row used to travel into the weights and stay there, every prediction answering nan from then on without a word. No file is needed for this
	_unmoved(t, "KNN answers what it answered before those two", knn, CLASS_TEST, func(): _bad_fits(knn, false))
	_unmoved(t, "LinReg predicts what it predicted before them", linreg, PROBE_LIN, func(): _bad_fits(linreg, true))
	_unmoved(t, "LogReg predicts what it predicted before them", logreg, CLASS_TEST, func(): _bad_fits(logreg, true))
	_unmoved(t, "SVM predicts what it predicted before them", svm, CLASS_TEST, func(): _bad_fits(svm, true))
	t.check_equal("a fit that goes through answers true", DTDAKNN.new(1).fit([[0.0], [1.0]], [5, 9]), true)
	t.check_empty("KNN _predict before _fit", DTDAKNN.new(3).predict([[1]]))
