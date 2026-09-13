# The runner itself, through its decision helpers, which answer a reason instead of touching the counters: an assertion that raises while it is being evaluated counts neither as a pass nor as a failure, it vanishes, and the last line stays green on a suite that ran fewer checks. A reason that raised answers null and null != "" is true, which is why _refused() asks for the type as well: without that, a guard dropped from _near_reason() reads here as a guard that held.

const PLAN = 23

func _refused(reason):
	return reason is String and reason != ""

func _run(t):
	t.section("Test runner, the guards that keep an assertion from vanishing")
	t.check_equal("an int is a number", t._is_number(2), true)
	t.check_equal("a float is a number", t._is_number(2.0), true)
	# a nan carries TYPE_FLOAT and answers false to every comparison, so a check that disqualifies by "too far apart" reads it as close enough
	t.check_equal("a nan is not a number", t._is_number(NAN), false)
	t.check_equal("close enough is no reason to fail", t._near_reason(1.0, 1.0001, 0.001), "")
	t.check("too far apart is a reason to fail", _refused(t._near_reason(1.0, 2.0, 0.001)))
	t.check("a null lands as a failure", _refused(t._near_reason(null, 1.0, 0.001)))
	t.check("an expected value that is not a number lands as a failure", _refused(t._near_reason(1.0, null, 0.001)))
	t.check("two infinities land as a failure", _refused(t._near_reason(INF, INF, 0.001)))
	t.check_equal("two equal arrays are no reason to fail", t._near_array_reason([1, 2], [1, 2], 0.001), "")
	t.check("two different arrays are a reason to fail", _refused(t._near_array_reason([1, 2], [1, 3], 0.001)))
	t.check("sizes that differ are a reason to fail", _refused(t._near_array_reason([1], [1, 2], 0.001)))
	t.check("a null array lands as a failure", _refused(t._near_array_reason(null, [1, 2], 0.001)))
	t.check("an expected value that is not an array lands as a failure", _refused(t._near_array_reason([1, 2], null, 0.001)))
	t.check("an array holding a null lands as a failure", _refused(t._near_array_reason([1, null], [1, 2], 0.001)))
	t.check("an array of two infinities lands as a failure", _refused(t._near_array_reason([INF], [INF], 0.001)))
	t.check_equal("a string compared to a number is a plain false", t._same("2", 2), false)
	t.check_equal("2 and 2.0 are still the same number", t._same(2, 2.0), true)
	t.check_equal("nested arrays still compare", t._same([[1], [2.0]], [[1.0], [2]]), true)
	t.check_equal("arrays that differ are not the same", t._same([1, 2], [1, 3]), false)
	t.check_equal("arrays of different sizes are not the same", t._same([1], [1, 2]), false)
	t.check_equal("dictionaries that differ are not the same", t._same({"a": 1}, {"a": 2}), false)
	t.check_equal("dictionaries of different sizes are not the same", t._same({"a": 1}, {"a": 1, "b": 2}), false)
	# compared with ==, not handed to check_equal(), which would ask _same() whether _same() answered right; unlike "not", a call that raised answers null here and null == false is false, which fails
	t.check("two different strings are not the same", t._same("a", "b") == false)
