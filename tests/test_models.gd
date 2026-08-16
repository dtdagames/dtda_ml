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
	t.check_near_array("features x1000 give the same classes",
		big_log._predict(_scale_rows(CLASS_TEST, 1000)), [0, 1, 1])

	t.section("SVM")
	var svm = DTDASVM.new(0.01, 0.01, 1000)
	svm._fit(X_log, y_log)
	t.check_near_array("predicts -1 and 1", svm._predict(CLASS_TEST), [-1, 1, 1])
	var big_svm = DTDASVM.new(0.01, 0.01, 1000)
	big_svm._fit(_scale_rows(X_log, 1000), y_log)
	t.check_near_array("features x1000 give the same classes",
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
	var knn_back = DTDAKNN.new(3)
	t.check("KNN loads", knn_back._load(knn_path))
	t.check_near_array("a reloaded KNN predicts the same", knn_back._predict(CLASS_TEST), knn._predict(CLASS_TEST))

	t.section("Persistence guards (the errors below are expected)")
	# a linear regression must refuse a file holding a KNN
	var wrong = DTDALinReg.new(0.01, 1000)
	t.check("_load refuses another kind of model", not wrong._load(knn_path))
	t.check("_load refuses a missing file", not DTDALinReg.new(0.01, 1000)._load("user://does_not_exist.json"))
	t.check("_save refuses a model that was never fitted", not DTDALinReg.new(0.01, 1000)._save(path))

	t.section("Fit guards (the errors below are expected)")
	t.check_empty("KNN _predict before _fit", DTDAKNN.new(3)._predict([[1]]))
	t.check_empty("LinReg _predict before _fit", DTDALinReg.new(0.01, 10)._predict([[1]]))
	t.check_empty("LogReg _predict before _fit", DTDALogReg.new(0.01, 10)._predict([[1]]))
	t.check_empty("SVM _predict before _fit", DTDASVM.new(0.01, 0.01, 10)._predict([[1]]))
