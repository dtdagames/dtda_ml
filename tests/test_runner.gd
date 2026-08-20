# The runner itself.
# An assertion that raises a script error while it is being evaluated is counted
# neither as a pass nor as a failure: it vanishes, and the final line stays green
# on a suite that ran fewer checks. The guards that keep that from happening are
# worth their own tests, so they are exercised here through the decision helpers,
# which answer a reason instead of touching the counters.

const PLAN = 34

func _run(t):
	# through a variable, so nothing is folded away before it runs
	var zero = 0.0
	var nan_value = zero / zero
	var inf_value = 1.0 / zero

	t.section("Test runner, what counts as a number")
	t.check("a null is not a number", not t._is_number(null))
	t.check("a string is not a number", not t._is_number("2"))
	t.check("an array is not a number", not t._is_number([2]))
	t.check("an int is a number", t._is_number(2))
	t.check("a float is a number", t._is_number(2.0))
	# a nan carries TYPE_FLOAT and answers false to every comparison, so a check that
	# disqualifies by "too far apart" would read it as close enough
	t.check("a nan is not a number", not t._is_number(nan_value))
	t.check("an infinity still is one", t._is_number(inf_value))

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

	t.check("a nan on the got side lands as a failure", t._near_reason(nan_value, 5.0, 0.001) != "")
	t.check("a nan on the expected side lands as a failure", t._near_reason(5.0, nan_value, 0.001) != "")
	# an infinity is a distance away from a number and no distance at all from another
	# infinity: the subtraction leaves a nan, which the comparison would let through
	t.check("an infinity against a number lands as a failure", t._near_reason(inf_value, 5.0, 0.001) != "")
	t.check("two infinities land as a failure", t._near_reason(inf_value, inf_value, 0.001) != "")

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
	t.check("an array holding a nan lands as a failure",
		t._near_array_reason([1.0, nan_value], [1.0, 5.0], 0.001) != "")
	t.check("an expected array holding a nan lands as a failure",
		t._near_array_reason([1.0, 5.0], [1.0, nan_value], 0.001) != "")
	# the array rule is the single one check_near() uses, applied element by element,
	# and this is what says so: a second copy of it would have to remember on its own
	# that two infinities leave a nan behind
	t.check("an array of two infinities lands as a failure",
		t._near_array_reason([inf_value], [inf_value], 0.001) != "")

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
	# _same() reads _is_number() too, and turning a nan away there moves nothing:
	# a nan was already equal to nothing, itself included
	t.check_equal("a nan is not equal to a number", t._same(nan_value, 5.0), false)
	t.check_equal("a nan is not even equal to a nan", t._same(nan_value, nan_value), false)
	t.check_equal("nested arrays still compare", t._same([[1], [2.0]], [[1.0], [2]]), true)
	# through a variable, "null is Array" alone does not even compile
	var nothing = null
	t.check("a null is not an empty array", not (nothing is Array))
