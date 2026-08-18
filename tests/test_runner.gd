# The runner itself.
# An assertion that raises a script error while it is being evaluated is counted
# neither as a pass nor as a failure: it vanishes, and the final line stays green
# on a suite that ran fewer checks. The guards that keep that from happening are
# worth their own tests, so they are exercised here through the decision helpers,
# which answer a reason instead of touching the counters.

const PLAN = 23

func _run(t):
	t.section("Test runner, what counts as a number")
	t.check("a null is not a number", not t._is_number(null))
	t.check("a string is not a number", not t._is_number("2"))
	t.check("an array is not a number", not t._is_number([2]))
	t.check("an int is a number", t._is_number(2))
	t.check("a float is a number", t._is_number(2.0))

	t.section("Test runner, check_near guards")
	# "" is what _near_reason() answers when the assertion holds
	t.check_equal("close enough is no reason to fail", t._near_reason(1.0, 1.0001, 0.001), "")
	t.check("too far apart is a reason to fail", t._near_reason(1.0, 2.0, 0.001) != "")
	# a model regressed to null is the failure _check_fitted() exists to produce:
	# it must land as a FAIL, not disappear inside abs()
	t.check("a null lands as a failure", t._near_reason(null, 1.0, 0.001) != "")
	t.check("a string lands as a failure", t._near_reason("1.0", 1.0, 0.001) != "")
	t.check("an expected value that is not a number lands as a failure",
		t._near_reason(1.0, null, 0.001) != "")

	t.section("Test runner, check_near_array guards")
	t.check_equal("two equal arrays are no reason to fail",
		t._near_array_reason([1, 2], [1, 2], 0.001), "")
	t.check("two different arrays are a reason to fail",
		t._near_array_reason([1, 2], [1, 3], 0.001) != "")
	t.check("sizes that differ are a reason to fail",
		t._near_array_reason([1], [1, 2], 0.001) != "")
	# the exact shape of a _predict() that came back empty or null
	t.check("a null array lands as a failure", t._near_array_reason(null, [1, 2], 0.001) != "")
	t.check("an array holding a null lands as a failure",
		t._near_array_reason([1, null], [1, 2], 0.001) != "")
	t.check("an array of strings lands as a failure",
		t._near_array_reason(["a"], [1], 0.001) != "")

	t.section("Test runner, check_equal guards")
	# "2" == 2 raises in Godot 4, so _same() has to sort the types out before
	# comparing. check_equal() against false, not "not _same(...)": a raise would
	# answer null, and "not null" is true, which would hide the problem
	t.check_equal("a null compared to a number is a plain false", t._same(null, 2), false)
	t.check_equal("a string compared to a number is a plain false", t._same("2", 2), false)
	t.check_equal("an array compared to a null is a plain false", t._same([1], null), false)
	t.check_equal("a dictionary compared to a string is a plain false", t._same({}, "x"), false)
	# an int and a float holding the same value stay comparable, they are numbers
	t.check_equal("2 and 2.0 are still the same number", t._same(2, 2.0), true)
	t.check_equal("nested arrays still compare", t._same([[1], [2.0]], [[1.0], [2]]), true)
	# through a variable, "null is Array" alone does not even compile
	var nothing = null
	t.check("a null is not an empty array", not (nothing is Array))
