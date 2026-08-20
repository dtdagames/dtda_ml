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
const PLAN = 79

# write a handmade file and hand it to a model, for the guards on the file itself
func _load_written(content, model):
	var path = "user://dtda_ml_test_handmade.json"
	var file = FileAccess.open(path, FileAccess.WRITE)
	file.store_string(content)
	file.close()
	return model.load(path)

func _run(t):
	var ml = DTDATools.new()

	t.section("KNN")
	var knn = DTDAKNN.new(3)
	knn.fit(ml.drop_variable(DATA_KNN, 6), ml.get_variable(DATA_KNN, 6))
	t.check_near_array("predicts the expected labels", knn.predict([
		[1, 4, 1, 1, 0, 0],
		[2, 2, 4, 1, 1, 1],
		[4, 1, 1, 0, 1, 0],
	]), [3, 6, 5])

	# the two nearest neighbours out of three carry label 1, the closest carries 0.
	# a 1-NN would answer 0, the majority vote answers 1
	var voter = DTDAKNN.new(3)
	voter.fit([[0.1], [0.5], [0.6]], [0, 1, 1])
	t.check_near_array("takes the majority, not the closest neighbour", voter.predict([[0.0]]), [1])

	# both rows only differ by their last feature, which the distance must take into account
	var full = DTDAKNN.new(1)
	full.fit([[0, 0], [0, 5]], [10, 20])
	t.check_near_array("the distance uses every feature", full.predict([[0, 5]]), [20])

	# asking for more neighbours than there are rows must not read out of bounds
	var greedy = DTDAKNN.new(10)
	greedy.fit([[0], [1]], [7, 8])
	t.check_equal("k larger than the training set", greedy.predict([[0]]).size(), 1)

	t.section("Linear Regression")
	var X_lin = ml.drop_variable(DATA_LINR, 1)
	var y_lin = ml.get_variable(DATA_LINR, 1)
	var linreg = DTDALinReg.new(0.01, 1000)
	linreg.fit(X_lin, y_lin)
	var lin_pred = linreg.predict(X_lin)
	t.check("fits the training set closely", ml.r2_score(lin_pred, y_lin) > 0.98)
	t.check("predictions stay in the unit of the target", lin_pred[0] > 10000.0 and lin_pred[0] < 200000.0)

	# standardization makes the model independent of the scale of the features
	var big = DTDALinReg.new(0.01, 1000)
	big.fit(_scale_rows(X_lin, 1000), y_lin)
	t.check_near_array("features x1000 give the same predictions",
		big.predict(_scale_rows([[7.2], [9.0], [11.1]], 1000)),
		linreg.predict([[7.2], [9.0], [11.1]]), 1.0)

	t.section("Logistic Regression")
	var X_log = ml.drop_variable(DATA_LOGR, 6)
	var y_log = ml.get_variable(DATA_LOGR, 6)
	var logreg = DTDALogReg.new(0.01, 1000)
	logreg.fit(X_log, y_log)
	t.check_near_array("separates the training set", logreg.predict(X_log), y_log)
	t.check_near_array("predicts the expected classes", logreg.predict(CLASS_TEST), [0, 1, 1])

	# without standardization exp() overflows here
	var big_log = DTDALogReg.new(0.01, 1000)
	big_log.fit(_scale_rows(X_log, 1000), y_log)
	t.check_near_array("LogReg on features x1000 gives the same classes",
		big_log.predict(_scale_rows(CLASS_TEST, 1000)), [0, 1, 1])

	t.section("SVM")
	var svm = DTDASVM.new(0.01, 0.01, 1000)
	svm.fit(X_log, y_log)
	t.check_near_array("predicts -1 and 1", svm.predict(CLASS_TEST), [-1, 1, 1])
	var big_svm = DTDASVM.new(0.01, 0.01, 1000)
	big_svm.fit(_scale_rows(X_log, 1000), y_log)
	t.check_near_array("SVM on features x1000 gives the same classes",
		big_svm.predict(_scale_rows(CLASS_TEST, 1000)), [-1, 1, 1])

	t.section("Saving and loading")
	var path = "user://dtda_ml_test_linreg.json"
	var before = linreg.predict([[7.2], [9.0], [11.1]])
	t.check("_save reports a success", linreg.save(path))
	var reloaded = DTDALinReg.new(0.01, 1000)
	t.check("_load reports a success", reloaded.load(path))
	t.check_near_array("a reloaded model predicts the same", reloaded.predict([[7.2], [9.0], [11.1]]), before, 0.001)

	var knn_path = "user://dtda_ml_test_knn.json"
	t.check("KNN saves", knn.save(knn_path))
	# built with a different count on purpose: a receiver already holding 3 would load
	# the same 3 whether _from_dict assigns it or not, and the field would go unpinned
	var knn_back = DTDAKNN.new(1)
	t.check("KNN loads", knn_back.load(knn_path))
	t.check_equal("a reloaded KNN takes the neighbour count from the file", knn_back.num_neighbors, 3)
	t.check_near_array("a reloaded KNN predicts the same", knn_back.predict(CLASS_TEST), knn.predict(CLASS_TEST))

	t.section("Persistence guards (the errors below are expected)")
	# a linear regression must refuse a file holding a KNN
	var wrong = DTDALinReg.new(0.01, 1000)
	# check_equal against false, not "not <call>": a call that raises a script error
	# answers null, which "not" reads as a success
	t.check_equal("_load refuses another kind of model", wrong.load(knn_path), false)
	t.check_equal("_load refuses a missing file", DTDALinReg.new(0.01, 1000).load("user://does_not_exist.json"), false)
	t.check_equal("_save refuses a model that was never fitted", DTDALinReg.new(0.01, 1000).save(path), false)

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

	var linreg_before = linreg.predict([[7.2], [9.0], [11.1]])
	t.check_equal("LinReg refuses a file whose scaler holds a zero scale",
		_load_written('{"model": "DTDALinReg", "version": 1, "rate": 0.5, "iterations": 3, "W": [9.9], "b": 7.7, ' + zero_x + '}',
			linreg), false)
	t.check_near_array("and LinReg predicts what it predicted before",
		linreg.predict([[7.2], [9.0], [11.1]]), linreg_before, 0.001)

	var logreg_before = logreg.predict(CLASS_TEST)
	t.check_equal("LogReg refuses a file whose scaler holds a zero scale",
		_load_written('{"model": "DTDALogReg", "version": 1, "rate": 0.5, "iterations": 3, "W": [9.9], "b": 7.7, ' + zero_s + '}',
			logreg), false)
	t.check_near_array("and LogReg predicts what it predicted before",
		logreg.predict(CLASS_TEST), logreg_before)

	var svm_before = svm.predict(CLASS_TEST)
	t.check_equal("SVM refuses a file whose scaler holds a zero scale",
		_load_written('{"model": "DTDASVM", "version": 1, "lr": 0.5, "lambda": 0.5, "iter": 3, "W": [9.9], "b": 7.7, ' + zero_s + '}',
			svm), false)
	t.check_near_array("and SVM predicts what it predicted before",
		svm.predict(CLASS_TEST), svm_before)
	# and no hyperparameter of the refused file either: each of the three was built
	# with a rate of 0.01 and 1000 rounds, the refused files carry 0.5 and 3
	t.check_equal("no scrap of the refused file is left behind",
		[linreg.rate, logreg.iterations, svm.iter], [0.01, 1000, 1000])

	t.section("Persistence, weights read out of a file (the errors below are expected)")
	# predict() computes with every one of these numbers, so a file that holds a text
	# or an empty list where the weights belong is not usable. It used to load: with a
	# text _load answered null after a cascade, with a list of the wrong shape it
	# answered true and the model was ruined either way
	t.check_equal("LinReg refuses weights that are not a list",
		_load_written('{"model": "DTDALinReg", "version": 1, "W": "nope", ' + '"x_scaler": {"mode": 0, "offsets": [0.0], "scales": [1.0]}, "y_scaler": {"mode": 0, "offsets": [0.0], "scales": [1.0]}' + '}', DTDALinReg.new(0.01, 1000)), false)
	t.check_equal("LinReg refuses an empty list of weights",
		_load_written('{"model": "DTDALinReg", "version": 1, "W": [], ' + '"x_scaler": {"mode": 0, "offsets": [0.0], "scales": [1.0]}, "y_scaler": {"mode": 0, "offsets": [0.0], "scales": [1.0]}' + '}', DTDALinReg.new(0.01, 1000)), false)
	t.check_equal("LinReg refuses weights holding something that is not a number",
		_load_written('{"model": "DTDALinReg", "version": 1, "W": [1.0, "nope"], ' + '"x_scaler": {"mode": 0, "offsets": [0.0], "scales": [1.0]}, "y_scaler": {"mode": 0, "offsets": [0.0], "scales": [1.0]}' + '}', DTDALinReg.new(0.01, 1000)), false)
	# the one way a number that is not finite reaches this far: a literal too large for
	# a float, which JSON hands back as an infinity. A literal nan or inf in the file
	# would make it unreadable and never get here at all
	t.check_equal("LinReg refuses a weight too large to hold",
		_load_written('{"model": "DTDALinReg", "version": 1, "W": [1e400], "b": 0.0, ' + '"x_scaler": {"mode": 0, "offsets": [0.0], "scales": [1.0]}, "y_scaler": {"mode": 0, "offsets": [0.0], "scales": [1.0]}' + '}', DTDALinReg.new(0.01, 1000)), false)
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
	var linreg_weights = linreg.predict([[7.2], [9.0], [11.1]])
	t.check_equal("a file with unusable weights is refused",
		_load_written('{"model": "DTDALinReg", "version": 1, "rate": 0.5, "iterations": 3, "W": ["nope"], "b": 7.7, ' + '"x_scaler": {"mode": 0, "offsets": [0.0], "scales": [1.0]}, "y_scaler": {"mode": 0, "offsets": [0.0], "scales": [1.0]}' + '}',
			linreg), false)
	t.check_near_array("and LinReg predicts what it predicted before that one",
		linreg.predict([[7.2], [9.0], [11.1]]), linreg_weights, 0.001)
	var logreg_weights = logreg.predict(CLASS_TEST)
	t.check_equal("a file with an unusable intercept is refused",
		_load_written('{"model": "DTDALogReg", "version": 1, "iterations": 3, "W": [9.9], "b": "nope", ' + '"scaler": {"mode": 0, "offsets": [0.0], "scales": [1.0]}' + '}',
			logreg), false)
	t.check_near_array("and LogReg predicts what it predicted before that one",
		logreg.predict(CLASS_TEST), logreg_weights)
	var svm_weights = svm.predict(CLASS_TEST)
	t.check_equal("a file with an empty list of weights is refused",
		_load_written('{"model": "DTDASVM", "version": 1, "iter": 3, "W": [], "b": 7.7, ' + '"scaler": {"mode": 0, "offsets": [0.0], "scales": [1.0]}' + '}',
			svm), false)
	t.check_near_array("and SVM predicts what it predicted before that one",
		svm.predict(CLASS_TEST), svm_weights)

	t.section("Persistence, a KNN training set read out of a file (the errors below are expected)")
	# a KNN answers with the rows it kept, so those rows have to be rows of numbers.
	# A training set that is a text used to load with a success and only fall apart at
	# the first prediction. The labels are left alone, a KNN answers them as they come
	var knn_before = knn.predict(CLASS_TEST)
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
	# the neighbour count is read at every prediction, not at fit(): a text used to
	# load with a success and answer null, a count of zero a list of nulls
	t.check_equal("KNN refuses a neighbour count that is not a number",
		_load_written('{"model": "DTDAKNN", "version": 1, "num_neighbors": "nope", "X": [[0]], "Y": [3]}', knn), false)
	t.check_equal("KNN refuses a neighbour count below one",
		_load_written('{"model": "DTDAKNN", "version": 1, "num_neighbors": 0, "X": [[0]], "Y": [3]}', knn), false)
	t.check_near_array("and none of those refusals moved it",
		knn.predict(CLASS_TEST), knn_before)

	t.section("A fit that is refused changes nothing (the errors below are expected)")
	# fit() is handed whatever the caller computed, and one unlucky division upstream
	# is enough: a nan in a row used to travel into the weights and stay there, every
	# prediction answering nan from then on without a word. No file is needed for this.
	# Four different faults per model, so the invariant does not hang on one of them
	var zero = 0.0
	var nan_row = [[1.0, 2.0], [2.0, 1.0], [8.0, zero / zero]]
	var inf_row = [[1.0, 2.0], [2.0, 1.0], [8.0, 1.0 / zero]]
	var text_row = [[1.0, 2.0], [2.0, 1.0], [8.0, "nope"]]
	var ragged = [[1.0, 2.0], [2.0], [8.0, 9.0]]
	var three = [0, 1, 1]
	var sound_rows = [[1.0, 2.0], [2.0, 1.0], [8.0, 9.0]]

	var knn_probe = [[1, 4, 1, 1, 0, 0], [2, 2, 4, 1, 1, 1], [4, 1, 1, 0, 1, 0]]
	var knn_fit_before = knn.predict(knn_probe)
	t.check_equal("KNN refuses a row holding a nan", knn.fit(nan_row, three), false)
	t.check_equal("KNN refuses rows of unequal widths", knn.fit(ragged, three), false)
	t.check_equal("KNN _fit refuses more rows than labels", knn.fit(sound_rows, [0]), false)
	knn.fit(inf_row, three)
	knn.fit(text_row, three)
	t.check_near_array("KNN answers what it answered before those four",
		knn.predict(knn_probe), knn_fit_before)
	# The other side of the same line, and the one the README puts first: a KNN only
	# hands a label back, so a label naming a class is not a fault and must not be
	# refused. Without this, tightening the three fits that do weigh their labels into
	# all seven would go through without an objection
	var named = DTDAKNN.new(1)
	t.check_equal("KNN takes labels that name a class",
		named.fit([[0.0, 0.0], [0.5, 0.5], [9.0, 9.0], [9.5, 9.5]], ["cave", "cave", "camp", "camp"]), true)
	t.check_equal("and hands the name back", named.predict([[0.2, 0.2], [9.2, 9.2]]), ["cave", "camp"])

	var lin_before = linreg.predict([[7.2], [9.0], [11.1]])
	t.check_equal("LinReg refuses a row holding a nan", linreg.fit(nan_row, three), false)
	linreg.fit(inf_row, three)
	linreg.fit(text_row, three)
	linreg.fit(ragged, three)
	# labels this one descends on, so they have to be numbers
	t.check_equal("LinReg refuses more rows than labels", linreg.fit(sound_rows, [0]), false)
	t.check_equal("LinReg refuses labels that are not numbers",
		linreg.fit([[1.0], [2.0]], ["red", "blue"]), false)
	t.check_near_array("LinReg predicts what it predicted before them",
		linreg.predict([[7.2], [9.0], [11.1]]), lin_before, 0.001)

	var log_before = logreg.predict(CLASS_TEST)
	t.check_equal("LogReg refuses a row holding an infinity", logreg.fit(inf_row, three), false)
	logreg.fit(nan_row, three)
	logreg.fit(text_row, three)
	logreg.fit(ragged, three)
	t.check_equal("LogReg refuses more rows than labels", logreg.fit(sound_rows, [0]), false)
	t.check_equal("LogReg refuses labels that are not numbers",
		logreg.fit([[1.0], [2.0]], ["red", "blue"]), false)
	t.check_near_array("LogReg predicts what it predicted before them",
		logreg.predict(CLASS_TEST), log_before)

	var svm_fit_before = svm.predict(CLASS_TEST)
	t.check_equal("SVM refuses a row that is not a row", svm.fit([[1.0, 2.0], "nope"], [0, 1]), false)
	svm.fit(nan_row, three)
	svm.fit(inf_row, three)
	svm.fit(ragged, three)
	t.check_equal("SVM refuses more rows than labels", svm.fit(sound_rows, [0]), false)
	t.check_equal("SVM refuses labels that are not numbers",
		svm.fit([[1.0], [2.0]], ["red", "blue"]), false)
	t.check_near_array("SVM predicts what it predicted before them",
		svm.predict(CLASS_TEST), svm_fit_before)

	# and a fit that goes through still says so
	t.check_equal("a fit that goes through answers true",
		DTDAKNN.new(1).fit([[0.0], [1.0]], [5, 9]), true)

	t.section("Fit guards (the errors below are expected)")
	t.check_empty("KNN _predict before _fit", DTDAKNN.new(3).predict([[1]]))
	t.check_empty("LinReg _predict before _fit", DTDALinReg.new(0.01, 10).predict([[1]]))
	t.check_empty("LogReg _predict before _fit", DTDALogReg.new(0.01, 10).predict([[1]]))
	t.check_empty("SVM _predict before _fit", DTDASVM.new(0.01, 0.01, 10).predict([[1]]))
