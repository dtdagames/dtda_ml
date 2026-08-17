# MLTools helpers, metrics and DTDAScaler.

# how many assertions this suite runs, checked by the runner
const PLAN = 43

func _run(t):
	var ml = MLTools.new()

	t.section("MLTools data helpers")
	var data = [
		[1, 2, 3],
		[4, 5, 6],
	]
	t.check_equal("_dropVariable removes the column", ml._dropVariable(data, 2), [[1, 2], [4, 5]])
	t.check_equal("_getVariable keeps the column", ml._getVariable(data, 2), [3, 6])
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
	t.check_near("_accuracy", ml._accuracy(y_pred, y_test), 33.33, 0.01)
	var confusion = ml._confusion_matrix(y_pred, y_test)
	t.check_equal("_confusion_matrix tp", confusion["tp"], 1)
	t.check_equal("_confusion_matrix fp", confusion["fp"], 1)
	t.check_equal("_confusion_matrix fn", confusion["fn"], 1)
	t.check_equal("_confusion_matrix tn", confusion["tn"], 0)
	t.check_near("_precision", ml._precision(y_pred, y_test), 0.5)
	t.check_near("_recall", ml._recall(y_pred, y_test), 0.5)
	t.check_near("_f1_score", ml._f1_score(y_pred, y_test), 0.5)
	t.check_near("_accuracy on a perfect prediction", ml._accuracy(y_test, y_test), 100.0)
	# nothing predicted positive, the denominator of the precision is zero
	t.check_near("_precision without any positive predicted", ml._precision([0, 0, 0], y_test), 0.0)

	t.section("Regression metrics")
	var truth = [10.0, 20.0, 30.0]
	t.check_near("_mse", ml._mse([12.0, 20.0, 30.0], truth), 4.0 / 3.0)
	t.check_near("_rmse", ml._rmse([12.0, 20.0, 30.0], truth), sqrt(4.0 / 3.0))
	t.check_near("_mae", ml._mae([12.0, 18.0, 30.0], truth), 4.0 / 3.0)
	t.check_near("_r2_score on a perfect fit", ml._r2_score(truth, truth), 1.0)
	# answering the mean everywhere is the definition of R2 = 0
	t.check_near("_r2_score of a constant model on the mean", ml._r2_score([20.0, 20.0, 20.0], truth), 0.0)
	# no variance to explain, must not divide by zero
	t.check_near("_r2_score on a constant target", ml._r2_score([1, 1, 1], [5, 5, 5]), 0.0)

	t.section("Metric guards (the errors below are expected)")
	t.check_near("_get_perf on an empty prediction", ml._get_perf([], [1], 0), 0.0)
	t.check_near("_get_perf on mismatched sizes", ml._get_perf([1, 0, 1], [0, 1], 0), 0.0)
	t.check_near("_accuracy on mismatched sizes", ml._accuracy([1, 0, 1], [0, 1]), 0.0)
	t.check_equal("_confusion_matrix on mismatched sizes", ml._confusion_matrix([1], [0, 1]), {})
	t.check_near("_mse on an empty prediction", ml._mse([], []), 0.0)

	t.section("DTDAScaler")
	var raw = [
		[1.0, 100.0],
		[3.0, 300.0],
		[5.0, 500.0],
	]
	var minmax = DTDAScaler.new(DTDAScaler.MINMAX)
	var scaled = minmax._fit_transform(raw)
	t.check_near_array("min-max first row", scaled[0], [0.0, 0.0])
	t.check_near_array("min-max middle row", scaled[1], [0.5, 0.5])
	t.check_near_array("min-max last row", scaled[2], [1.0, 1.0])
	var restored = minmax._inverse_transform(scaled)
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
	var int_scaled = int_minmax._fit_transform(integers)
	t.check_near_array("min-max on an integer column", int_scaled[1], [40000.0 / 81000.0])
	t.check_near_array("_inverse_transform on an integer column",
		int_minmax._inverse_transform(int_scaled)[1], [80000.0])

	var standard = DTDAScaler.new()
	var centered = standard._fit_transform(raw)
	var first_column = ml._getVariable(centered, 0)
	t.check_near("standardized column has a null mean", ml._mean_array(first_column), 0.0)
	t.check_near("standardized column has a unit deviation", ml._std_array(first_column), 1.0)

	# a constant column must not divide by zero
	var constant = DTDAScaler.new()
	var flat = constant._fit_transform([[7.0], [7.0]])
	t.check_near_array("a constant column stays finite", flat[0], [0.0])

	# the scaling learned on the training set must apply as is to new data
	var reused = DTDAScaler.new(DTDAScaler.MINMAX)
	reused._fit(raw)
	t.check_near_array("_transform reuses the learned scaling", reused._transform([[5.0, 500.0]])[0], [1.0, 1.0])

	t.section("DTDAScaler guard (the error below is expected)")
	t.check_empty("_transform before _fit returns an empty array", DTDAScaler.new()._transform([[1.0]]))
