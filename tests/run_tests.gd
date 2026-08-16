extends SceneTree

# Headless test runner.
#   godot --headless --script res://tests/run_tests.gd
# Exits with 0 when everything passes, 1 otherwise, which is what CI reads.
#
# Some tests exercise the guards of the addon on purpose, so the output
# contains expected "MLTools: ..." errors. They are not failures: only the
# FAIL lines and the final count are.

const TEST_SCRIPTS = [
	"res://tests/test_tools.gd",
	"res://tests/test_models.gd",
	"res://tests/test_tree.gd",
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
		script.new()._run(self)

	print("")
	print("%d passed, %d failed" % [passed, failed])
	quit(1 if failed > 0 else 0)

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
func _same(a, b):
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

# floats never compare exactly, so every numeric check goes through a tolerance
func check_near(name, got, expected, tolerance = 0.0001):
	if abs(got - expected) <= tolerance:
		_pass()
	else:
		_fail(name, "got %s, expected %s (+/- %s)" % [got, expected, tolerance])

func check_near_array(name, got, expected, tolerance = 0.0001):
	if got.size() != expected.size():
		_fail(name, "got %d values, expected %d" % [got.size(), expected.size()])
		return
	for i in got.size():
		if abs(got[i] - expected[i]) > tolerance:
			_fail(name, "got %s, expected %s (+/- %s)" % [got, expected, tolerance])
			return
	_pass()

func check_empty(name, got):
	if got is Array and got.size() == 0:
		_pass()
	else:
		_fail(name, "expected an empty array, got %s" % [got])
