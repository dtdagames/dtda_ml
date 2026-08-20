# MLTools helpers, metrics and DTDAScaler.

# how many assertions this suite runs, checked by the runner
const PLAN = 82

func _run(t):
	var ml = DTDATools.new()

	t.section("MLTools data helpers")
	var data = [
		[1, 2, 3],
		[4, 5, 6],
	]
	t.check_equal("_dropVariable removes the column", ml.drop_variable(data, 2), [[1, 2], [4, 5]])
	t.check_equal("_getVariable keeps the column", ml.get_variable(data, 2), [3, 6])
	t.check_near("_mean_array", ml._mean_array([1, 2, 3, 4]), 2.5)
	t.check_near("_std_array", ml._std_array([2, 4, 4, 4, 5, 5, 7, 9]), 2.0)
	t.check_near("_std_array returns 1.0 on a constant column", ml._std_array([7, 7, 7]), 1.0)
	t.check_equal("_column_to_matrix", ml._column_to_matrix([1, 2]), [[1], [2]])
	t.check_equal("_matrix_to_column", ml._matrix_to_column([[1], [2]]), [1, 2])
	# dividing integers must keep the decimal part, 1 / 2 is 0.5 and not 0
	t.check_near_array("_divide_array_coef on integers", ml._divide_array_coef([1, 3], 2), [0.5, 1.5])
	t.check_near_array("_divide_inverse_array_coef on integers", ml._divide_inverse_array_coef([2, 4], 1), [0.5, 0.25])

	t.section("MLTools sign")
	# a value of exactly 0 must land on 1, class 0 does not exist in a -1/1 model
	t.check_equal("_sign_array sends 0 to 1", ml._sign_array([2.0, -2.0, 0.0]), [1, -1, 1])

	t.section("Classification metrics")
	# expected [0, 1, 1] against [1, 1, 0] gives one tp, one fp, one fn
	var y_test = [0, 1, 1]
	var y_pred = [1, 1, 0]
	t.check_near("_accuracy", ml.accuracy(y_pred, y_test), 33.33, 0.01)
	var confusion = ml.confusion_matrix(y_pred, y_test)
	t.check_equal("_confusion_matrix tp", confusion["tp"], 1)
	t.check_equal("_confusion_matrix fp", confusion["fp"], 1)
	t.check_equal("_confusion_matrix fn", confusion["fn"], 1)
	t.check_equal("_confusion_matrix tn", confusion["tn"], 0)
	t.check_near("_precision", ml.precision(y_pred, y_test), 0.5)
	t.check_near("_recall", ml.recall(y_pred, y_test), 0.5)
	t.check_near("_f1_score", ml.f1_score(y_pred, y_test), 0.5)
	t.check_near("_accuracy on a perfect prediction", ml.accuracy(y_test, y_test), 100.0)
	# nothing predicted positive, the denominator of the precision is zero
	t.check_near("_precision without any positive predicted", ml.precision([0, 0, 0], y_test), 0.0)

	t.section("Regression metrics")
	var truth = [10.0, 20.0, 30.0]
	t.check_near("_mse", ml.mse([12.0, 20.0, 30.0], truth), 4.0 / 3.0)
	t.check_near("_rmse", ml.rmse([12.0, 20.0, 30.0], truth), sqrt(4.0 / 3.0))
	t.check_near("_mae", ml.mae([12.0, 18.0, 30.0], truth), 4.0 / 3.0)
	t.check_near("_r2_score on a perfect fit", ml.r2_score(truth, truth), 1.0)
	# answering the mean everywhere is the definition of R2 = 0
	t.check_near("_r2_score of a constant model on the mean", ml.r2_score([20.0, 20.0, 20.0], truth), 0.0)
	# no variance to explain, must not divide by zero
	t.check_near("_r2_score on a constant target", ml.r2_score([1, 1, 1], [5, 5, 5]), 0.0)

	t.section("Metric guards (the errors below are expected)")
	t.check_near("_get_perf on an empty prediction", ml.get_perf([], [1], 0), 0.0)
	t.check_near("_get_perf on mismatched sizes", ml.get_perf([1, 0, 1], [0, 1], 0), 0.0)
	t.check_near("_accuracy on mismatched sizes", ml.accuracy([1, 0, 1], [0, 1]), 0.0)
	t.check_equal("_confusion_matrix on mismatched sizes", ml.confusion_matrix([1], [0, 1]), {})
	t.check_near("_mse on an empty prediction", ml.mse([], []), 0.0)

	t.section("DTDAScaler")
	var raw = [
		[1.0, 100.0],
		[3.0, 300.0],
		[5.0, 500.0],
	]
	var minmax = DTDAScaler.new(DTDAScaler.MINMAX)
	var scaled = minmax.fit_transform(raw)
	t.check_near_array("min-max first row", scaled[0], [0.0, 0.0])
	t.check_near_array("min-max middle row", scaled[1], [0.5, 0.5])
	t.check_near_array("min-max last row", scaled[2], [1.0, 1.0])
	var restored = minmax.inverse_transform(scaled)
	t.check_near_array("_inverse_transform restores the first row", restored[0], raw[0])
	t.check_near_array("_inverse_transform restores the last row", restored[2], raw[2])

	# a column of integers must not trigger an integer division. 40000 / 81000
	# came out as 0 until the scaler forced its offset and scale to floats,
	# and the float literals above were not enough to catch it
	var integers = [
		[40000],
		[80000],
		[121000],
	]
	var int_minmax = DTDAScaler.new(DTDAScaler.MINMAX)
	var int_scaled = int_minmax.fit_transform(integers)
	t.check_near_array("min-max on an integer column", int_scaled[1], [40000.0 / 81000.0])
	t.check_near_array("_inverse_transform on an integer column",
		int_minmax.inverse_transform(int_scaled)[1], [80000.0])

	var standard = DTDAScaler.new()
	var centered = standard.fit_transform(raw)
	var first_column = ml.get_variable(centered, 0)
	t.check_near("standardized column has a null mean", ml._mean_array(first_column), 0.0)
	t.check_near("standardized column has a unit deviation", ml._std_array(first_column), 1.0)

	# a constant column must not divide by zero
	var constant = DTDAScaler.new()
	var flat = constant.fit_transform([[7.0], [7.0]])
	t.check_near_array("a constant column stays finite", flat[0], [0.0])

	# the scaling learned on the training set must apply as is to new data
	var reused = DTDAScaler.new(DTDAScaler.MINMAX)
	reused.fit(raw)
	t.check_near_array("_transform reuses the learned scaling", reused.transform([[5.0, 500.0]])[0], [1.0, 1.0])

	t.section("MLTools, numbers read out of a file")
	# what a model reads out of user:// has to be usable, not merely present
	t.check_equal("a list of numbers is a list of numbers",
		ml._check_number_array([1, 2.5], "M", "weights"), true)
	t.check_equal("a text is not a list", ml._check_number_array("nope", "M", "weights"), false)
	t.check_equal("a null is not a list either", ml._check_number_array(null, "M", "weights"), false)
	t.check_equal("an empty list holds no number", ml._check_number_array([], "M", "weights"), false)
	t.check_equal("a list holding a text is not a list of numbers",
		ml._check_number_array([1.0, "nope"], "M", "weights"), false)
	t.check_equal("a list holding a list is not one either",
		ml._check_number_array([1.0, [2.0]], "M", "weights"), false)
	t.check_equal("_check_number takes an int", ml._check_number(3, "M", "intercept"), true)
	t.check_equal("_check_number takes a float", ml._check_number(3.5, "M", "intercept"), true)
	t.check_equal("_check_number refuses a text", ml._check_number("3.5", "M", "intercept"), false)
	t.check_equal("_check_number refuses a null", ml._check_number(null, "M", "intercept"), false)

	# through variables, so nothing is folded away before it runs
	var zero = 0.0
	var nan_value = zero / zero
	var inf_value = 1.0 / zero
	t.check_equal("a nan is not a number to compute with",
		ml._check_number(nan_value, "M", "intercept"), false)
	t.check_equal("an infinity is not one either",
		ml._check_number(inf_value, "M", "intercept"), false)
	t.check_equal("a list holding a nan is not a list of numbers",
		ml._check_number_array([1.0, nan_value], "M", "weights"), false)
	t.check_equal("a list holding an infinity is not one either",
		ml._check_number_array([1.0, inf_value], "M", "weights"), false)

	t.section("MLTools, rows handed to a fit")
	# what a caller passes to fit() arrives from its own arithmetic, so one unlucky
	# division upstream is all it takes
	t.check_equal("a sound matrix is a sound matrix",
		ml._check_matrix([[1.0, 2.0], [3, 4]], "M"), true)
	t.check_equal("a matrix that is not a list", ml._check_matrix("nope", "M"), false)
	t.check_equal("a matrix with no rows", ml._check_matrix([], "M"), false)
	t.check_equal("a row that is not a list", ml._check_matrix([[1.0], "nope"], "M"), false)
	t.check_equal("a row with nothing in it", ml._check_matrix([[1.0], []], "M"), false)
	t.check_equal("a row holding a text", ml._check_matrix([[1.0], ["nope"]], "M"), false)
	t.check_equal("a row holding a nan", ml._check_matrix([[1.0], [nan_value]], "M"), false)
	t.check_equal("a row holding an infinity", ml._check_matrix([[1.0], [inf_value]], "M"), false)
	t.check_equal("rows of unequal widths", ml._check_matrix([[1.0, 2.0], [3.0]], "M"), false)
	# and the labels, counted rather than read: a label can be whatever names a class
	t.check_equal("as many labels as rows", ml._check_labels([[1.0], [2.0]], [7, 9], "M"), true)
	t.check_equal("labels that name a class rather than a number",
		ml._check_labels([[1.0], [2.0]], ["red", "blue"], "M"), true)
	t.check_equal("labels that are not a list", ml._check_labels([[1.0]], "nope", "M"), false)
	t.check_equal("fewer labels than rows", ml._check_labels([[1.0], [2.0]], [7], "M"), false)

	t.section("DTDAScaler, reading a saved scaler")
	# a scaler is written inside the file of the model that owns it, and that file
	# lives in user:// where it can be edited by hand. What is read back has to be
	# usable, not merely present: transform() reads an offset and a scale per column
	# and divides by the scale
	var sound = DTDAScaler.new()
	t.check_equal("a sound scaler is read back",
		sound.from_dict({"mode": DTDAScaler.MINMAX, "offsets": [1.0], "scales": [2.0]}), true)
	t.check_near_array("and scales with what it read", sound.transform([[5.0]])[0], [2.0])
	# the same trap as the feature index of a tree: a mode read back from JSON is a
	# float, and this one is compared against an enum
	var moded = DTDAScaler.new()
	moded.from_dict({"mode": 1.0, "offsets": [1.0], "scales": [2.0]})
	t.check_equal("the mode comes back as an integer", typeof(moded.mode), TYPE_INT)

	t.section("DTDAScaler, refusing a saved scaler (the errors below are expected)")
	t.check_equal("offsets that are not a list",
		DTDAScaler.new().from_dict({"offsets": "nope", "scales": [1.0]}), false)
	t.check_equal("scales that are not a list",
		DTDAScaler.new().from_dict({"offsets": [1.0], "scales": "nope"}), false)
	t.check_equal("a scaler with nothing in it",
		DTDAScaler.new().from_dict({"offsets": [], "scales": []}), false)
	t.check_equal("more offsets than scales",
		DTDAScaler.new().from_dict({"offsets": [1.0, 2.0], "scales": [1.0]}), false)
	t.check_equal("an offset that is not a number",
		DTDAScaler.new().from_dict({"offsets": [1.0, "nope"], "scales": [1.0, 2.0]}), false)
	# "2.5" and not "nope": float("nope") is 0.0, so the guard on the zero below would
	# answer for it and this assertion would not name the guard it claims
	t.check_equal("a scale that is not a number",
		DTDAScaler.new().from_dict({"offsets": [1.0, 2.0], "scales": [1.0, "2.5"]}), false)
	# this one used to load and answer inf at the first prediction, without an error
	t.check_equal("a scale of zero, which _transform would divide by",
		DTDAScaler.new().from_dict({"offsets": [1.0], "scales": [0.0]}), false)
	# and a refused dictionary must not take the standing scaler down with it
	var standing = DTDAScaler.new()
	standing.fit([[0.0], [10.0]])
	var standing_before = standing.transform([[5.0]])
	standing.from_dict({"offsets": [99.0], "scales": [0.0]})
	t.check_near_array("a refused scaler leaves the standing one alone",
		standing.transform([[5.0]])[0], standing_before[0])

	t.section("DTDAScaler guard (the error below is expected)")
	# the same answer as the eight models: false when it refused, true when it fitted
	t.check_equal("fit with no data answers false", DTDAScaler.new().fit([]), false)
	t.check_empty("_transform before _fit returns an empty array", DTDAScaler.new().transform([[1.0]]))
