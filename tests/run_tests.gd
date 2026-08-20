extends SceneTree

# Headless test runner.
#   godot --headless --script res://tests/run_tests.gd
# Exits with 0 when everything passes, 1 otherwise, which is what CI reads.
#
# Some tests exercise the guards of the addon on purpose, so the output contains
# expected "MLTools: ..." errors. Those ones are not failures.
#
# Everything else the engine prints is worth reading, because a GDScript runtime
# error raised while an assertion was being evaluated used to swallow that
# assertion whole: it counted neither as a pass nor as a failure, and the last
# line stayed green while the suite quietly ran fewer checks than it used to.
# Two things guard against that now and both must stay:
#   - check_near() and check_near_array() refuse a value that is not a number
#     instead of letting abs() raise, so a _predict() regressed to null lands as
#     a FAIL rather than disappearing
#   - every suite declares a PLAN, the number of assertions it runs, and the
#     runner fails when it records a different number
# So: the FAIL lines and the final count are what matter, and they can now be
# trusted to be complete.

const TEST_SCRIPTS = [
	"res://tests/test_runner.gd",
	"res://tests/test_tools.gd",
	"res://tests/test_models.gd",
	"res://tests/test_tree.gd",
	"res://tests/test_qlearning.gd",
	"res://tests/test_forest.gd",
	"res://tests/test_kmeans.gd",
]

var passed = 0
var failed = 0

func _initialize():
	for path in TEST_SCRIPTS:
		var script = load(path)
		if script == null:
			failed += 1
			print("FAIL  cannot load %s" % path)
			continue
		var before = passed + failed
		script.new()._run(self)
		_check_plan(path, script, passed + failed - before)

	print("")
	print("%d passed, %d failed" % [passed, failed])
	quit(1 if failed > 0 else 0)

# a suite stopping halfway, on a script error for instance, records fewer
# assertions than it announces. Without this the count would simply come out
# smaller, which nobody reads as a failure
func _check_plan(path, script, recorded):
	var constants = script.get_script_constant_map()
	if not constants.has("PLAN"):
		failed += 1
		print("FAIL  %s declares no PLAN" % path)
		return
	if recorded != constants["PLAN"]:
		failed += 1
		print("FAIL  %s ran %d assertions, its PLAN announces %d" % [path, recorded, constants["PLAN"]])

func section(title):
	print("")
	print("== %s" % title)

func _pass():
	passed += 1

func _fail(name, detail):
	failed += 1
	print("FAIL  %s" % name)
	print("        %s" % detail)

func check(name, condition):
	if condition:
		_pass()
	else:
		_fail(name, "expected true")

# deep comparison, so nested arrays and dictionaries do not depend on how a
# given Godot version implements ==
# some pairs of types raise on ==, "2" == 2 for one, which would print a script
# error and answer null. Numbers are compared across int and float, everything
# else has to match in type first, so this is safe on whatever a broken model
# hands back
func _same(a, b):
	if _is_number(a) and _is_number(b):
		return a == b
	if typeof(a) != typeof(b):
		return false
	if a is Array and b is Array:
		if a.size() != b.size():
			return false
		for i in a.size():
			if not _same(a[i], b[i]):
				return false
		return true
	if a is Dictionary and b is Dictionary:
		if a.size() != b.size():
			return false
		for key in a:
			if not b.has(key) or not _same(a[key], b[key]):
				return false
		return true
	return a == b

func check_equal(name, got, expected):
	if _same(got, expected):
		_pass()
	else:
		_fail(name, "got %s, expected %s" % [got, expected])

func _is_number(value):
	return typeof(value) == TYPE_INT or typeof(value) == TYPE_FLOAT

# "" when the two values are close enough, the reason of the failure otherwise.
# the rule lives apart from check_near() so it can be tested without going
# through the counters. The type check is the point: abs(null - 1.0) raises,
# and a raise inside an assertion makes it vanish from the count
func _near_reason(got, expected, tolerance):
	if not _is_number(got) or not _is_number(expected):
		return "got %s, expected a number near %s" % [got, expected]
	if abs(got - expected) > tolerance:
		return "got %s, expected %s (+/- %s)" % [got, expected, tolerance]
	return ""

# floats never compare exactly, so every numeric check goes through a tolerance
func check_near(name, got, expected, tolerance = 0.0001):
	var reason = _near_reason(got, expected, tolerance)
	if reason == "":
		_pass()
	else:
		_fail(name, reason)

func _near_array_reason(got, expected, tolerance):
	if not (got is Array) or not (expected is Array):
		return "got %s, expected an array of numbers like %s" % [got, expected]
	if got.size() != expected.size():
		return "got %d values, expected %d" % [got.size(), expected.size()]
	for i in got.size():
		if not _is_number(got[i]) or not _is_number(expected[i]):
			return "got %s, expected an array of numbers like %s" % [got, expected]
		if abs(got[i] - expected[i]) > tolerance:
			return "got %s, expected %s (+/- %s)" % [got, expected, tolerance]
	return ""

func check_near_array(name, got, expected, tolerance = 0.0001):
	var reason = _near_array_reason(got, expected, tolerance)
	if reason == "":
		_pass()
	else:
		_fail(name, reason)

func check_empty(name, got):
	# "got is Array" is already false on a null, nothing can raise here
	if got is Array and got.size() == 0:
		_pass()
	else:
		_fail(name, "expected an empty array, got %s" % [got])
