const PLAN = 48

func _run(t):
	var ml = DTDATools.new()
	# through variables, so nothing is folded away before it runs
	var zero = 0.0
	var nan_value = zero / zero
	var inf_value = 1.0 / zero

	t.section("MLTools helpers")
	t.check_equal("_dropVariable removes the column", ml.drop_variable([[1, 2, 3], [4, 5, 6]], 2), [[1, 2], [4, 5]])
	t.check_equal("_getVariable keeps the column", ml.get_variable([[1, 2, 3], [4, 5, 6]], 2), [3, 6])
	t.check_near("_mean_array", ml._mean_array([1, 2, 3, 4]), 2.5)
	t.check_near("_std_array", ml._std_array([2, 4, 4, 4, 5, 5, 7, 9]), 2.0)
	t.check_near_array("an empty column has a mean of 0 and a deviation of 1, which stays safe to divide by", [ml._mean_array([]), ml._std_array([])], [0.0, 1.0])
	t.check_equal("_column_to_matrix wraps", ml._column_to_matrix([1, 2]), [[1], [2]])
	t.check_equal("_matrix_to_column unwraps", ml._matrix_to_column([[1], [2]]), [1, 2])
	# integer arguments, not float literals: dividing two integers must keep the decimal part, 1 / 2 is 0.5 and not 0
	t.check_near_array("_divide_array_coef on integers", ml._divide_array_coef([1, 3], 2), [0.5, 1.5])
	t.check_near_array("_divide_inverse_array_coef on integers", ml._divide_inverse_array_coef([2, 4], 1), [0.5, 0.25])
	# a value of exactly 0 must land on 1, class 0 does not exist in a -1/1 model
	t.check_equal("_sign_array sends 0 to 1", ml._sign_array([2.0, -2.0, 0.0]), [1, -1, 1])

	t.section("Metrics")
	var y_test = [0, 1, 1]
	var y_pred = [1, 1, 0]
	t.check_near("_accuracy", ml.accuracy(y_pred, y_test), 33.33, 0.01)
	t.check_equal("_confusion_matrix counts around the positive label", ml.confusion_matrix(y_pred, y_test), {"tp": 1, "fp": 1, "fn": 1, "tn": 0})
	t.check_near("_precision", ml.precision(y_pred, y_test), 0.5)
	t.check_near("_recall", ml.recall(y_pred, y_test), 0.5)
	t.check_near("_f1_score", ml.f1_score(y_pred, y_test), 0.5)
	t.check_near("_precision without any positive predicted", ml.precision([0, 0, 0], y_test), 0.0)
	t.check_near("_f1_score with nothing positive on either side", ml.f1_score([0, 0], [0, 0]), 0.0)
	t.check_near_array("_get_perf turns a regression and an SVM into classes before counting", [ml.get_perf([0.9, 0.2, 0.8], [1, 0, 0], 1), ml.get_perf([1, -1], [1, 0], 3)], [66.67, 100.0], 0.01)
	var truth = [10.0, 20.0, 30.0]
	t.check_near("_mse", ml.mse([12.0, 20.0, 30.0], truth), 4.0 / 3.0)
	t.check_near("_rmse", ml.rmse([12.0, 20.0, 30.0], truth), sqrt(4.0 / 3.0))
	t.check_near("_mae", ml.mae([12.0, 18.0, 30.0], truth), 4.0 / 3.0)
	t.check_near("_r2_score on a perfect fit", ml.r2_score(truth, truth), 1.0)
	t.check_near("_r2_score of a constant model on the mean", ml.r2_score([20.0, 20.0, 20.0], truth), 0.0)
	t.check_near("_r2_score on a constant target, with no variance to explain", ml.r2_score([1, 1, 1], [5, 5, 5]), 0.0)

	t.section("Metric guards (the errors below are expected)")
	t.check_near_array("every metric refuses an empty pair or two sizes that do not match", [ml.get_perf([], [1], 0), ml.get_perf([1, 0, 1], [0, 1], 0), ml.accuracy([1, 0, 1], [0, 1]), ml.mae([1.0], [1.0, 2.0]), ml.mse([], []), ml.precision([1, 0, 1], [0, 1]), ml.recall([1, 0, 1], [0, 1]), ml.f1_score([1, 0, 1], [0, 1])], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
	t.check_equal("_confusion_matrix refuses them too", ml.confusion_matrix([1], [0, 1]), {})

	t.section("DTDAScaler")
	var raw = [[1.0, 100.0], [3.0, 300.0], [5.0, 500.0]]
	var minmax = DTDAScaler.new(DTDAScaler.MINMAX)
	var scaled = minmax.fit_transform(raw)
	t.check_near_array("min-max brings every column into [0, 1]", scaled[0] + scaled[1] + scaled[2], [0.0, 0.0, 0.5, 0.5, 1.0, 1.0])
	t.check_near_array("_fit_transform then _inverse_transform gives the data back in its own unit", minmax.inverse_transform(scaled)[0] + minmax.inverse_transform(scaled)[2], [1.0, 100.0, 5.0, 500.0])
	t.check_near_array("_transform reuses the learned scaling on a row that was not in the training set", minmax.transform([[6.0, 600.0]])[0], [1.25, 1.25])
	# 40000 / 81000 came out as 0 until fit() forced the offset and the scale of a MINMAX column to floats. The value below falls only when both casts go, either one on its own promotes the division; the types are what answers for each of them alone
	var int_minmax = DTDAScaler.new(DTDAScaler.MINMAX)
	var int_scaled = int_minmax.fit_transform([[40000], [80000], [121000]])
	t.check_near_array("min-max on a column of integers keeps the decimal part", int_scaled[1], [40000.0 / 81000.0])
	t.check_equal("and leaves neither an integer offset nor an integer scale behind", [typeof(int_minmax.offsets[0]), typeof(int_minmax.scales[0])], [TYPE_FLOAT, TYPE_FLOAT])
	var standardized = ml.get_variable(DTDAScaler.new().fit_transform(raw), 0)
	t.check_near("standardizing gives a column a null mean", ml._mean_array(standardized), 0.0)
	t.check_near("and a unit deviation", ml._std_array(standardized), 1.0)
	t.check_near_array("a constant column is never divided by zero, in either mode", [DTDAScaler.new().fit_transform([[7.0], [7.0]])[0][0], DTDAScaler.new(DTDAScaler.MINMAX).fit_transform([[7.0], [7.0]])[0][0]], [0.0, 0.0])
	# check_equal against false rather than "not <call>": a GDScript error answers null, and "not null" would turn a crash into a pass
	t.check_equal("fit with no data answers false, and transform before a fit has nothing to say", [DTDAScaler.new().fit([]), DTDAScaler.new().transform([[1.0]])], [false, []])

	t.section("DTDAScaler, reading a saved scaler (the errors below are expected)")
	var sound = DTDAScaler.new()
	t.check_equal("a sound scaler is read back, and scales with what it read", [sound.from_dict({"mode": DTDAScaler.MINMAX, "offsets": [1.0], "scales": [2.0]}), sound.transform([[5.0]])[0]], [true, [2.0]])
	var moded = DTDAScaler.new()
	moded.from_dict({"mode": 1.0, "offsets": [1.0], "scales": [2.0]})
	# a mode read back from JSON is a float, and this one is compared against an enum
	t.check_equal("the mode comes back as an integer", typeof(moded.mode), TYPE_INT)
	t.check_equal("a saved scaler that is not two lists of the same length, or holds nothing at all, is refused", [DTDAScaler.new().from_dict({"offsets": "nope", "scales": [1.0]}), DTDAScaler.new().from_dict({"offsets": [1.0], "scales": "nope"}), DTDAScaler.new().from_dict({"offsets": [], "scales": []}), DTDAScaler.new().from_dict({"offsets": [1.0, 2.0], "scales": [1.0]})], [false, false, false, false])
	# "2.5" and not "nope": float("nope") is 0.0, so the guard on the zero below would answer for it and this would not name the guard it claims
	t.check_equal("an offset or a scale that is not a number is refused", [DTDAScaler.new().from_dict({"offsets": [1.0, "nope"], "scales": [1.0, 2.0]}), DTDAScaler.new().from_dict({"offsets": [1.0, 2.0], "scales": [1.0, "2.5"]})], [false, false])
	var standing = DTDAScaler.new()
	standing.fit([[0.0], [10.0]])
	# this one used to load and answer inf at the first prediction, without a word
	t.check_equal("a scale of zero, which transform() would divide by, is refused", standing.from_dict({"offsets": [99.0], "scales": [0.0]}), false)
	t.check_near_array("and a refused scaler leaves the standing one exactly as it was", standing.transform([[7.5]])[0], [0.5])

	t.section("MLTools, numbers read out of a file (the errors below are expected)")
	t.check_equal("a list of numbers is a list of numbers, and a lone number is one", [ml._check_number_array([1, 2.5], "M", "weights"), ml._check_number(3, "M", "intercept"), ml._check_number(3.5, "M", "intercept")], [true, true, true])
	t.check_equal("a text, a null, a nan and an infinity are not numbers to compute with", [ml._check_number("3.5", "M", "intercept"), ml._check_number(null, "M", "intercept"), ml._check_number(nan_value, "M", "intercept"), ml._check_number(inf_value, "M", "intercept")], [false, false, false, false])
	t.check_equal("a list that is not a list, or holds nothing at all", [ml._check_number_array("nope", "M", "weights"), ml._check_number_array(null, "M", "weights"), ml._check_number_array([], "M", "weights")], [false, false, false])
	t.check_equal("a list holding a text, a list, a nan or an infinity is not a list of numbers", [ml._check_number_array([1.0, "nope"], "M", "weights"), ml._check_number_array([1.0, [2.0]], "M", "weights"), ml._check_number_array([1.0, nan_value], "M", "weights"), ml._check_number_array([1.0, inf_value], "M", "weights")], [false, false, false, false])

	t.section("MLTools, rows handed to a fit (the errors below are expected)")
	t.check_equal("a sound matrix, and as many labels as rows whatever they name", [ml._check_matrix([[1.0, 2.0], [3, 4]], "M"), ml._check_labels([[1.0], [2.0]], [7, 9], "M"), ml._check_labels([[1.0], [2.0]], ["red", "blue"], "M")], [true, true, true])
	t.check_equal("a matrix that is not a list, one with no rows, a row that is not a list, an empty row, a row holding a nan", [ml._check_matrix("nope", "M"), ml._check_matrix([], "M"), ml._check_matrix([[1.0], "nope"], "M"), ml._check_matrix([[1.0], []], "M"), ml._check_matrix([[1.0], [nan_value]], "M")], [false, false, false, false, false])
	t.check_equal("rows of unequal widths, labels that are not a list, fewer labels than rows", [ml._check_matrix([[1.0, 2.0], [3.0]], "M"), ml._check_labels([[1.0]], "nope", "M"), ml._check_labels([[1.0], [2.0]], [7], "M")], [false, false, false])
