extends SceneTree

# Measures the two proportion rules of CLAUDE.md and says where the repository
# stands. It is not part of the test suite and declares no PLAN.
#   godot --headless --script res://tools/check_proportions.gd
# Exits 1 when a threshold is crossed, 0 otherwise.

const COMMENT_CAP = 0.10
const TEST_CAP = 0.25

# this script is the ruler, not a thing being measured: out of both reports
const SELF_PATH = "res://tools/check_proportions.gd"

# the runner is a runner, not a test file: the comment cap applies to it, the
# test cap does not
const RUNNER_PATH = "res://tests/run_tests.gd"

const MODULES = {
	"knn": "res://addons/dtda_ml/models/dtda_ml_knn.gd",
	"linreg": "res://addons/dtda_ml/models/dtda_ml_linreg.gd",
	"logreg": "res://addons/dtda_ml/models/dtda_ml_logreg.gd",
	"svm": "res://addons/dtda_ml/models/dtda_ml_svm.gd",
	"tree": "res://addons/dtda_ml/models/dtda_ml_tree.gd",
	"forest": "res://addons/dtda_ml/models/dtda_ml_forest.gd",
	"kmeans": "res://addons/dtda_ml/models/dtda_ml_kmeans.gd",
	"qlearning": "res://addons/dtda_ml/models/dtda_ml_qlearning.gd",
	"dtda_ml_tools": "res://addons/dtda_ml/dtda_ml_tools.gd",
	"dtda_ml_scaler": "res://addons/dtda_ml/dtda_ml_scaler.gd",
	"dtda_ml_tools_compat": "res://addons/dtda_ml/dtda_ml_tools_compat.gd",
	"run_tests": RUNNER_PATH,
}

# what each suite exercises. A test file is allowed a quarter of the non-empty
# lines of the modules listed here, and of nothing else.
const COVERAGE = {
	"res://tests/test_models.gd": ["knn", "linreg", "logreg", "svm"],
	"res://tests/test_tree.gd": ["tree"],
	"res://tests/test_forest.gd": ["forest"],
	"res://tests/test_kmeans.gd": ["kmeans"],
	"res://tests/test_qlearning.gd": ["qlearning"],
	"res://tests/test_tools.gd": ["dtda_ml_tools", "dtda_ml_scaler"],
	"res://tests/test_names.gd": [
		"dtda_ml_tools_compat",
		"knn", "linreg", "logreg", "svm", "tree", "forest", "kmeans", "qlearning",
		"dtda_ml_tools",
	],
	"res://tests/test_stepped.gd": [
		"dtda_ml_tools", "linreg", "logreg", "svm", "kmeans", "forest",
	],
	"res://tests/test_runner.gd": ["run_tests"],
}

const SKIP_DIRS = [".git", ".godot", ".claude", ".github"]

# what a line can be counted as
const BLANK = "blank"
const CODE = "code"
const COMMENT = "comment"
const TRAILING = "trailing"

# the counter has to tell a comment from a hash living inside a string, because
# GDScript puts hashes in strings: Color("#14161c"), a maze row, a format. A
# naive counter credits the demos with comments they do not have, and a counter
# that answers "no comment" everywhere looks exactly like a clean tree. So these
# lines are classified on every run, one by one, and a single wrong answer stops
# the report. What one line cannot show is in SEQUENCES below.
const PROBES = [
	["var total = 0", CODE],
	["	# a full line comment", COMMENT],
	["	var total = 0  # a trailing comment", TRAILING],
	["	var tint = Color(\"#14161c\")", CODE],
	["	var tint = Color(\"#14161c\")  # the panel background", TRAILING],
	["	if GRID[y][x] == \"#\":", CODE],
	["	var row = '...##......G'", CODE],
	["	var row = '#top' # and a real one", TRAILING],
	["	var quoted = \"a \\\" then # not a comment\"", CODE],
	["", BLANK],
	["		", BLANK],
	["#!/usr/bin/env", COMMENT],
]

# a triple-quoted string outlives the line that opens it, and so does the state
# the counter carries. Each sequence below is classified in order through one
# single state, never reset between its lines: that carry is the only thing
# standing between a hash inside a multi-line string and a comment, and no file
# in the tree opens one today, so nothing but these pairs would notice it break.
const SEQUENCES = [
	["a triple-quoted string spanning lines", [
		["	var doc = \"\"\"", CODE],
		["	# this hash lives inside that string, it opens no comment", CODE],
		["	\"\"\"  # this one does", TRAILING],
	]],
	["a single-quoted string left open stops at its own line", [
		["	var broken = \"unterminated", CODE],
		["	# so the next line still opens a real comment", COMMENT],
	]],
]

var over_budget = 0

func _initialize():
	if not _self_check():
		print("")
		print("the line counter misclassifies, the figures below would be worth nothing")
		quit(1)
		return
	var files = []
	_collect("res://", files)
	files.sort()
	_report_comments(files)
	_report_tests()
	print("")
	if over_budget > 0:
		print("%d file(s) over budget" % over_budget)
	else:
		print("every file within budget")
	quit(1 if over_budget > 0 else 0)

# --- the line counter -------------------------------------------------------

# index of the first hash that opens a real comment, -1 when the line carries
# none. state["triple"] holds the multi-line string delimiter left open by the
# previous line, so a hash inside one is not read as a comment either.
func _comment_index(line, state):
	var quote = state["triple"]
	var i = 0
	var n = line.length()
	while i < n:
		var c = line[i]
		if quote != "":
			if c == "\\":
				i += 2
				continue
			if quote.length() == 3:
				if line.substr(i, 3) == quote:
					quote = ""
					i += 3
					continue
			elif c == quote:
				quote = ""
				i += 1
				continue
			i += 1
			continue
		if c == "#":
			# a comment swallows the rest of the line, nothing stays open past it
			state["triple"] = ""
			return i
		if c == "\"" or c == "'":
			var triple = c + c + c
			if line.substr(i, 3) == triple:
				quote = triple
				i += 3
			else:
				quote = c
				i += 1
			continue
		i += 1
	# a single-quoted string left open is a syntax error, not something that
	# carries to the next line. Only a triple-quoted one carries.
	state["triple"] = quote if quote.length() == 3 else ""
	return -1

func _classify(line, state):
	var at = _comment_index(line, state)
	if at < 0:
		return BLANK if line.strip_edges() == "" else CODE
	return COMMENT if line.substr(0, at).strip_edges() == "" else TRAILING

func _measure(path):
	var counts = {"non_empty": 0, "comments": 0, "full": 0, "trailing": 0, "total": 0}
	var file = FileAccess.open(path, FileAccess.READ)
	if file == null:
		print("  cannot read %s" % path)
		return counts
	var text = file.get_as_text()
	file.close()
	var state = {"triple": ""}
	for raw in text.replace("\r\n", "\n").replace("\r", "\n").split("\n"):
		counts["total"] += 1
		var kind = _classify(raw, state)
		if kind == BLANK:
			continue
		counts["non_empty"] += 1
		if kind == COMMENT:
			counts["full"] += 1
			counts["comments"] += 1
		elif kind == TRAILING:
			counts["trailing"] += 1
			counts["comments"] += 1
	return counts

func _report_probe(probe, state):
	var got = _classify(probe[0], state)
	var shown = probe[0].replace("	", "    ")
	if shown.strip_edges() == "":
		shown = "(blank line)"
	if got != probe[1]:
		print("  MISREAD   %-50s -> %s, expected %s" % [shown, got, probe[1]])
		return false
	print("  %-9s %s" % [got, shown])
	return true

func _self_check():
	print("== Line counter, what it tells apart")
	var ok = true
	for probe in PROBES:
		# a fresh state per line: these cases are about one line alone
		if _report_probe(probe, {"triple": ""}) == false:
			ok = false
	for sequence in SEQUENCES:
		print("  -- %s, one state carried through the lines below" % sequence[0])
		var state = {"triple": ""}
		for probe in sequence[1]:
			if _report_probe(probe, state) == false:
				ok = false
	return ok

# --- the two reports --------------------------------------------------------

func _collect(dir_path, into):
	var dir = DirAccess.open(dir_path)
	if dir == null:
		return
	dir.list_dir_begin()
	var entry = dir.get_next()
	while entry != "":
		var full = dir_path.path_join(entry)
		if dir.current_is_dir():
			if not (entry in SKIP_DIRS):
				_collect(full, into)
		elif entry.ends_with(".gd") and full != SELF_PATH:
			into.append(full)
		entry = dir.get_next()
	dir.list_dir_end()

func _report_comments(files):
	print("")
	print("== Comments, at most %d%% of the non-empty lines of a .gd file" % int(COMMENT_CAP * 100))
	print("  %-44s %9s %8s %8s %7s  %s" % ["file", "non-empty", "full", "trailing", "pct", "verdict"])
	var total_lines = 0
	var total_comments = 0
	for path in files:
		var counts = _measure(path)
		total_lines += counts["non_empty"]
		total_comments += counts["comments"]
		var pct = 0.0
		if counts["non_empty"] > 0:
			pct = 100.0 * float(counts["comments"]) / float(counts["non_empty"])
		var allowed = int(floor(COMMENT_CAP * float(counts["non_empty"])))
		var flag = ""
		if counts["comments"] > allowed:
			flag = "OVER by %d line(s)" % (counts["comments"] - allowed)
			over_budget += 1
		print("  %-44s %9d %8d %8d %6.1f%%  %s" % [
			path.replace("res://", ""),
			counts["non_empty"], counts["full"], counts["trailing"], pct, flag,
		])
	var whole = 0.0
	if total_lines > 0:
		whole = 100.0 * float(total_comments) / float(total_lines)
	print("  %-44s %9d %8d %8s %6.1f%%" % ["ALL .gd", total_lines, total_comments, "", whole])

func _report_tests():
	print("")
	print("== Tests, at most %d%% of the non-empty lines of the modules a suite covers" % int(TEST_CAP * 100))
	print("  %-28s %7s %9s %8s  %s" % ["test file", "lines", "modules", "budget", "verdict"])
	var paths = COVERAGE.keys()
	paths.sort()
	var total_test = 0
	var total_budget = 0
	for path in paths:
		var test_lines = _measure(path)["non_empty"]
		var module_lines = 0
		for key in COVERAGE[path]:
			module_lines += _measure(MODULES[key])["non_empty"]
		var budget = int(floor(TEST_CAP * float(module_lines)))
		total_test += test_lines
		total_budget += budget
		var flag = ""
		if test_lines > budget:
			flag = "OVER by %d line(s)" % (test_lines - budget)
			over_budget += 1
		print("  %-28s %7d %9d %8d  %s" % [
			path.replace("res://tests/", ""), test_lines, module_lines, budget, flag,
		])
	var verdict = ""
	if total_test > total_budget:
		verdict = "over by %d line(s)" % (total_test - total_budget)
	print("  %-28s %7d %9s %8d  %s" % ["ALL suites", total_test, "", total_budget, verdict])
