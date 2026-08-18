# KNN, Linear Regression, Logistic Regression and SVM.

const DATA_KNN = [
	[2, 4, 2, 1, 0, 0, 3],
	[2, 2, 4, 0, 0, 0, 4],
	[4, 2, 1, 1, 0, 1, 5],
	[2, 2, 4, 0, 1, 1, 6],
]

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

const CLASS_TEST = [
	[1, 3, 1, 0, 1, 0],
	[2, 2, 4, 1, 1, 1],
	[4, 1, 1, 0, 1, 0],
]

# multiply every feature by a factor, to check the models are scale independent
func _scale_rows(rows, factor):
	var scaled = []
	for i in rows.size():
		scaled.push_back([])
		for u in rows[i].size():
			scaled[i].push_back(rows[i][u] * factor)
	return scaled

# how many assertions this suite runs, checked by the runner
const PLAN = 59

# write a handmade file and hand it to a model, for the guards on the file itself
func _load_written(content, model):
	var path = "user://dtda_ml_test_handmade.json"
	var file = FileAccess.open(path, FileAccess.WRITE)
	file.store_string(content)
	file.close()
	return model._load(path)

func _run(t):
	var ml = MLTools.new()

	t.section("KNN")
	var knn = DTDAKNN.new(3)
	knn._fit(ml._dropVariable(DATA_KNN, 6), ml._getVariable(DATA_KNN, 6))
	t.check_near_array("predicts the expected labels", knn._predict([
		[1, 4, 1, 1, 0, 0],
		[2, 2, 4, 1, 1, 1],
		[4, 1, 1, 0, 1, 0],
	]), [3, 6, 5])

	# the two nearest neighbours out of three carry label 1, the closest carries 0.
	# a 1-NN would answer 0, the majority vote answers 1
	var voter = DTDAKNN.new(3)
	voter._fit([[0.1], [0.5], [0.6]], [0, 1, 1])
	t.check_near_array("takes the majority, not the closest neighbour", voter._predict([[0.0]]), [1])

	# both rows only differ by their last feature, which the distance must take into account
	var full = DTDAKNN.new(1)
	full._fit([[0, 0], [0, 5]], [10, 20])
	t.check_near_array("the distance uses every feature", full._predict([[0, 5]]), [20])

	# asking for more neighbours than there are rows must not read out of bounds
	var greedy = DTDAKNN.new(10)
	greedy._fit([[0], [1]], [7, 8])
	t.check_equal("k larger than the training set", greedy._predict([[0]]).size(), 1)

	t.section("Linear Regression")
	var X_lin = ml._dropVariable(DATA_LINR, 1)
	var y_lin = ml._getVariable(DATA_LINR, 1)
	var linreg = DTDALinReg.new(0.01, 1000)
	linreg._fit(X_lin, y_lin)
	var lin_pred = linreg._predict(X_lin)
	t.check("fits the training set closely", ml._r2_score(lin_pred, y_lin) > 0.98)
	t.check("predictions stay in the unit of the target", lin_pred[0] > 10000.0 and lin_pred[0] < 200000.0)

	# standardization makes the model independent of the scale of the features
	var big = DTDALinReg.new(0.01, 1000)
	big._fit(_scale_rows(X_lin, 1000), y_lin)
	t.check_near_array("features x1000 give the same predictions",
		big._predict(_scale_rows([[7.2], [9.0], [11.1]], 1000)),
		linreg._predict([[7.2], [9.0], [11.1]]), 1.0)

	t.section("Logistic Regression")
	var X_log = ml._dropVariable(DATA_LOGR, 6)
	var y_log = ml._getVariable(DATA_LOGR, 6)
	var logreg = DTDALogReg.new(0.01, 1000)
	logreg._fit(X_log, y_log)
	t.check_near_array("separates the training set", logreg._predict(X_log), y_log)
	t.check_near_array("predicts the expected classes", logreg._predict(CLASS_TEST), [0, 1, 1])

	# without standardization exp() overflows here
	var big_log = DTDALogReg.new(0.01, 1000)
	big_log._fit(_scale_rows(X_log, 1000), y_log)
	t.check_near_array("LogReg on features x1000 gives the same classes",
		big_log._predict(_scale_rows(CLASS_TEST, 1000)), [0, 1, 1])

	t.section("SVM")
	var svm = DTDASVM.new(0.01, 0.01, 1000)
	svm._fit(X_log, y_log)
	t.check_near_array("predicts -1 and 1", svm._predict(CLASS_TEST), [-1, 1, 1])
	var big_svm = DTDASVM.new(0.01, 0.01, 1000)
	big_svm._fit(_scale_rows(X_log, 1000), y_log)
	t.check_near_array("SVM on features x1000 gives the same classes",
		big_svm._predict(_scale_rows(CLASS_TEST, 1000)), [-1, 1, 1])

	t.section("Saving and loading")
	var path = "user://dtda_ml_test_linreg.json"
	var before = linreg._predict([[7.2], [9.0], [11.1]])
	t.check("_save reports a success", linreg._save(path))
	var reloaded = DTDALinReg.new(0.01, 1000)
	t.check("_load reports a success", reloaded._load(path))
	t.check_near_array("a reloaded model predicts the same", reloaded._predict([[7.2], [9.0], [11.1]]), before, 0.001)

	var knn_path = "user://dtda_ml_test_knn.json"
	t.check("KNN saves", knn._save(knn_path))
	# built with a different count on purpose: a receiver already holding 3 would load
	# the same 3 whether _from_dict assigns it or not, and the field would go unpinned
	var knn_back = DTDAKNN.new(1)
	t.check("KNN loads", knn_back._load(knn_path))
	t.check_equal("a reloaded KNN takes the neighbour count from the file", knn_back.num_neighbors, 3)
	t.check_near_array("a reloaded KNN predicts the same", knn_back._predict(CLASS_TEST), knn._predict(CLASS_TEST))

	t.section("Persistence guards (the errors below are expected)")
	# a linear regression must refuse a file holding a KNN
	var wrong = DTDALinReg.new(0.01, 1000)
	# check_equal against false, not "not <call>": a call that raises a script error
	# answers null, which "not" reads as a success
	t.check_equal("_load refuses another kind of model", wrong._load(knn_path), false)
	t.check_equal("_load refuses a missing file", DTDALinReg.new(0.01, 1000)._load("user://does_not_exist.json"), false)
	t.check_equal("_save refuses a model that was never fitted", DTDALinReg.new(0.01, 1000)._save(path), false)

	# The KNN file just refused is a KNN through and through, so what turns it away is
	# the shape of what it holds, not the name it carries: the assertion above says
	# nothing about _check_model_name(). Each file below is one its model could read
	# from end to end, and wrong on the "model" field alone
	t.check_equal("DTDAKNN refuses a file that only lies about its model name",
		_load_written('{"model": "NotAKNN", "version": 1, "num_neighbors": 1, "X": [[0]], "Y": [1]}',
			DTDAKNN.new(3)), false)
	t.check_equal("DTDALinReg refuses a file that only lies about its model name",
		_load_written('{"model": "NotALinReg", "version": 1, "rate": 0.01, "iterations": 10, "W": [1.0], "b": 0.0, "x_scaler": {"mode": 0, "offsets": [0.0], "scales": [1.0]}, "y_scaler": {"mode": 0, "offsets": [0.0], "scales": [1.0]}}',
			DTDALinReg.new(0.01, 1000)), false)
	t.check_equal("DTDALogReg refuses a file that only lies about its model name",
		_load_written('{"model": "NotALogReg", "version": 1, "rate": 0.01, "iterations": 10, "W": [1.0], "b": 0.0, "scaler": {"mode": 0, "offsets": [0.0], "scales": [1.0]}}',
			DTDALogReg.new(0.01, 1000)), false)
	t.check_equal("DTDASVM refuses a file that only lies about its model name",
		_load_written('{"model": "NotASVM", "version": 1, "lr": 0.01, "lambda": 0.01, "iter": 10, "W": [1.0], "b": 0.0, "scaler": {"mode": 0, "offsets": [0.0], "scales": [1.0]}}',
			DTDASVM.new(0.01, 0.01, 1000)), false)

	t.section("Persistence, a refused file changes nothing (the errors below are expected)")
	# A scaler that holds a zero divides by it at the first prediction. That file used
	# to load with a success and answer inf, and the three models below wrote the
	# weights of a file they went on to refuse over the ones they were working with.
	var zero_x = '"x_scaler": {"mode": 0, "offsets": [0.0], "scales": [0.0]}, "y_scaler": {"mode": 0, "offsets": [0.0], "scales": [1.0]}'
	var zero_s = '"scaler": {"mode": 0, "offsets": [0.0], "scales": [0.0]}'

	var linreg_before = linreg._predict([[7.2], [9.0], [11.1]])
	t.check_equal("LinReg refuses a file whose scaler holds a zero scale",
		_load_written('{"model": "DTDALinReg", "version": 1, "rate": 0.5, "iterations": 3, "W": [9.9], "b": 7.7, ' + zero_x + '}',
			linreg), false)
	t.check_near_array("and LinReg predicts what it predicted before",
		linreg._predict([[7.2], [9.0], [11.1]]), linreg_before, 0.001)

	var logreg_before = logreg._predict(CLASS_TEST)
	t.check_equal("LogReg refuses a file whose scaler holds a zero scale",
		_load_written('{"model": "DTDALogReg", "version": 1, "rate": 0.5, "iterations": 3, "W": [9.9], "b": 7.7, ' + zero_s + '}',
			logreg), false)
	t.check_near_array("and LogReg predicts what it predicted before",
		logreg._predict(CLASS_TEST), logreg_before)

	var svm_before = svm._predict(CLASS_TEST)
	t.check_equal("SVM refuses a file whose scaler holds a zero scale",
		_load_written('{"model": "DTDASVM", "version": 1, "lr": 0.5, "lambda": 0.5, "iter": 3, "W": [9.9], "b": 7.7, ' + zero_s + '}',
			svm), false)
	t.check_near_array("and SVM predicts what it predicted before",
		svm._predict(CLASS_TEST), svm_before)
	# and no hyperparameter of the refused file either: each of the three was built
	# with a rate of 0.01 and 1000 rounds, the refused files carry 0.5 and 3
	t.check_equal("no scrap of the refused file is left behind",
		[linreg.rate, logreg.iterations, svm.iter], [0.01, 1000, 1000])

	t.section("Persistence, weights read out of a file (the errors below are expected)")
	# _predict() computes with every one of these numbers, so a file that holds a text
	# or an empty list where the weights belong is not usable. It used to load: with a
	# text _load answered null after a cascade, with a list of the wrong shape it
	# answered true and the model was ruined either way
	t.check_equal("LinReg refuses weights that are not a list",
		_load_written('{"model": "DTDALinReg", "version": 1, "W": "nope", ' + '"x_scaler": {"mode": 0, "offsets": [0.0], "scales": [1.0]}, "y_scaler": {"mode": 0, "offsets": [0.0], "scales": [1.0]}' + '}', DTDALinReg.new(0.01, 1000)), false)
	t.check_equal("LinReg refuses an empty list of weights",
		_load_written('{"model": "DTDALinReg", "version": 1, "W": [], ' + '"x_scaler": {"mode": 0, "offsets": [0.0], "scales": [1.0]}, "y_scaler": {"mode": 0, "offsets": [0.0], "scales": [1.0]}' + '}', DTDALinReg.new(0.01, 1000)), false)
	t.check_equal("LinReg refuses weights holding something that is not a number",
		_load_written('{"model": "DTDALinReg", "version": 1, "W": [1.0, "nope"], ' + '"x_scaler": {"mode": 0, "offsets": [0.0], "scales": [1.0]}, "y_scaler": {"mode": 0, "offsets": [0.0], "scales": [1.0]}' + '}', DTDALinReg.new(0.01, 1000)), false)
	t.check_equal("LinReg refuses an intercept that is not a number",
		_load_written('{"model": "DTDALinReg", "version": 1, "W": [1.0], "b": "nope", ' + '"x_scaler": {"mode": 0, "offsets": [0.0], "scales": [1.0]}, "y_scaler": {"mode": 0, "offsets": [0.0], "scales": [1.0]}' + '}', DTDALinReg.new(0.01, 1000)), false)
	t.check_equal("LogReg refuses weights that are not a list",
		_load_written('{"model": "DTDALogReg", "version": 1, "W": {"a": 1}, ' + '"scaler": {"mode": 0, "offsets": [0.0], "scales": [1.0]}' + '}', DTDALogReg.new(0.01, 1000)), false)
	t.check_equal("LogReg refuses an intercept that is not a number",
		_load_written('{"model": "DTDALogReg", "version": 1, "W": [1.0], "b": "nope", ' + '"scaler": {"mode": 0, "offsets": [0.0], "scales": [1.0]}' + '}', DTDALogReg.new(0.01, 1000)), false)
	t.check_equal("SVM refuses weights that are not a list",
		_load_written('{"model": "DTDASVM", "version": 1, "W": 5, ' + '"scaler": {"mode": 0, "offsets": [0.0], "scales": [1.0]}' + '}', DTDASVM.new(0.01, 0.01, 1000)), false)
	t.check_equal("SVM refuses an intercept that is not a number",
		_load_written('{"model": "DTDASVM", "version": 1, "W": [1.0], "b": "nope", ' + '"scaler": {"mode": 0, "offsets": [0.0], "scales": [1.0]}' + '}', DTDASVM.new(0.01, 0.01, 1000)), false)

	t.section("Persistence, a refused file changes nothing (the errors below are expected)")
	# The invariant, and it must not hang on one door. The files below carry a sound
	# scaler and are refused for their weights alone, where the ones further down are
	# refused for a scaler holding a zero. Both have to leave the model untouched
	var linreg_weights = linreg._predict([[7.2], [9.0], [11.1]])
	t.check_equal("a file with unusable weights is refused",
		_load_written('{"model": "DTDALinReg", "version": 1, "rate": 0.5, "iterations": 3, "W": ["nope"], "b": 7.7, ' + '"x_scaler": {"mode": 0, "offsets": [0.0], "scales": [1.0]}, "y_scaler": {"mode": 0, "offsets": [0.0], "scales": [1.0]}' + '}',
			linreg), false)
	t.check_near_array("and LinReg predicts what it predicted before that one",
		linreg._predict([[7.2], [9.0], [11.1]]), linreg_weights, 0.001)
	var logreg_weights = logreg._predict(CLASS_TEST)
	t.check_equal("a file with an unusable intercept is refused",
		_load_written('{"model": "DTDALogReg", "version": 1, "iterations": 3, "W": [9.9], "b": "nope", ' + '"scaler": {"mode": 0, "offsets": [0.0], "scales": [1.0]}' + '}',
			logreg), false)
	t.check_near_array("and LogReg predicts what it predicted before that one",
		logreg._predict(CLASS_TEST), logreg_weights)
	var svm_weights = svm._predict(CLASS_TEST)
	t.check_equal("a file with an empty list of weights is refused",
		_load_written('{"model": "DTDASVM", "version": 1, "iter": 3, "W": [], "b": 7.7, ' + '"scaler": {"mode": 0, "offsets": [0.0], "scales": [1.0]}' + '}',
			svm), false)
	t.check_near_array("and SVM predicts what it predicted before that one",
		svm._predict(CLASS_TEST), svm_weights)

	t.section("Persistence, a KNN training set read out of a file (the errors below are expected)")
	# a KNN answers with the rows it kept, so those rows have to be rows of numbers.
	# A training set that is a text used to load with a success and only fall apart at
	# the first prediction. The labels are left alone, a KNN answers them as they come
	var knn_before = knn._predict(CLASS_TEST)
	# two files rather than one holding two texts: with both fields wrong, either half
	# of the check answers for the other and neither assertion names its own guard
	t.check_equal("KNN refuses training rows that are not a list",
		_load_written('{"model": "DTDAKNN", "version": 1, "num_neighbors": 1, "X": "nope", "Y": [3]}', knn), false)
	t.check_equal("KNN refuses labels that are not a list",
		_load_written('{"model": "DTDAKNN", "version": 1, "num_neighbors": 1, "X": [[0]], "Y": "nope"}', knn), false)
	t.check_equal("KNN refuses an empty training set",
		_load_written('{"model": "DTDAKNN", "version": 1, "num_neighbors": 1, "X": [], "Y": []}', knn), false)
	t.check_equal("KNN refuses more rows than labels",
		_load_written('{"model": "DTDAKNN", "version": 1, "num_neighbors": 1, "X": [[0], [1]], "Y": [3]}', knn), false)
	t.check_equal("KNN refuses a row that is not a row of numbers",
		_load_written('{"model": "DTDAKNN", "version": 1, "num_neighbors": 1, "X": [[0], ["nope"]], "Y": [3, 4]}', knn), false)
	# the neighbour count is read at every prediction, not at _fit(): a text used to
	# load with a success and answer null, a count of zero a list of nulls
	t.check_equal("KNN refuses a neighbour count that is not a number",
		_load_written('{"model": "DTDAKNN", "version": 1, "num_neighbors": "nope", "X": [[0]], "Y": [3]}', knn), false)
	t.check_equal("KNN refuses a neighbour count below one",
		_load_written('{"model": "DTDAKNN", "version": 1, "num_neighbors": 0, "X": [[0]], "Y": [3]}', knn), false)
	t.check_near_array("and none of those refusals moved it",
		knn._predict(CLASS_TEST), knn_before)

	t.section("Fit guards (the errors below are expected)")
	t.check_empty("KNN _predict before _fit", DTDAKNN.new(3)._predict([[1]]))
	t.check_empty("LinReg _predict before _fit", DTDALinReg.new(0.01, 10)._predict([[1]]))
	t.check_empty("LogReg _predict before _fit", DTDALogReg.new(0.01, 10)._predict([[1]]))
	t.check_empty("SVM _predict before _fit", DTDASVM.new(0.01, 0.01, 10)._predict([[1]]))
