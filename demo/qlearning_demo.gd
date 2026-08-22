extends Node2D

# The playable demo: two runs of the same agent, side by side, looping forever.
# Left, its very first episode, before it knows anything. Right, the policy it learned
# from four hundred of them. Nothing is narrated - the contrast is the whole point, and
# it is visible in about ten seconds.
#
# The learning itself happens once in _ready() and takes a few milliseconds; what runs
# per frame is only a replay of two recorded paths.

const CELL := 34
const MARGIN := 28
const TOP := 126
const GAP := 56
const CURVE_H := 96
const STEPS_PER_SECOND := 9.0
const EPISODES := 400
const SEED := 12345

const BACKGROUND := Color("14161c")
const FLOOR := Color("232733")
const LAVA := Color("b3402f")
const GOAL := Color("46a758")
const AGENT_LOST := Color("d9a441")
const AGENT_TAUGHT := Color("4a90d9")
const TRAIL := Color("3d4a61")
const TEXT := Color("c8cdd8")
const TEXT_DIM := Color("6b7385")

var world: DTDADemoMaze
var first_path: Array = []
var learned_path: Array = []
var lengths: Array = []
var elapsed := 0.0

func _ready() -> void:
	world = DTDADemoMaze.new()
	var trained := world.train(EPISODES, SEED)
	first_path = trained["first"]["path"]
	learned_path = trained["learned"]["path"]
	lengths = trained["lengths"]

func _process(delta: float) -> void:
	elapsed += delta
	queue_redraw()

# How far along its own path each side is. They loop independently, so the trained
# agent arrives again and again while the untrained one is still wandering - which
# says more than a caption would.
func _cursor(path: Array) -> int:
	if path.size() < 2:
		return 0
	var total := int(elapsed * STEPS_PER_SECOND)
	# a pause at the end of each loop, long enough to see where it stopped
	var period := path.size() + 8
	return min(total % period, path.size() - 1)

func _panel_origin(index: int) -> Vector2:
	return Vector2(MARGIN + index * (world.width * CELL + GAP), TOP)

func _draw() -> void:
	if world == null:
		return
	var font := ThemeDB.fallback_font
	# the whole viewport, not a computed box: a background sized by hand leaves a bar
	# of the engine grey along whichever edge the arithmetic got wrong
	draw_rect(get_viewport_rect(), BACKGROUND)

	draw_string(font, Vector2(MARGIN, 44), "A Q-Learning agent, before and after",
		HORIZONTAL_ALIGNMENT_LEFT, -1, 24, TEXT)
	draw_string(font, Vector2(MARGIN, 68),
		"It is not told where the goal is, only what a step is worth.",
		HORIZONTAL_ALIGNMENT_LEFT, -1, 15, TEXT_DIM)

	_draw_panel(0, first_path, AGENT_LOST, "Episode 1", "knows nothing")
	_draw_panel(1, learned_path, AGENT_TAUGHT, "After %d episodes" % EPISODES,
		"%d steps, the shortest there is" % (learned_path.size() - 1))
	_draw_curve(font)

func _draw_panel(index: int, path: Array, agent_color: Color, title: String, subtitle: String) -> void:
	var font := ThemeDB.fallback_font
	var origin := _panel_origin(index)
	var cursor := _cursor(path)

	draw_string(font, origin + Vector2(0, -26), title, HORIZONTAL_ALIGNMENT_LEFT, -1, 17, TEXT)
	draw_string(font, origin + Vector2(0, -8), subtitle, HORIZONTAL_ALIGNMENT_LEFT, -1, 13, TEXT_DIM)

	for y in world.height:
		for x in world.width:
			var cell := Vector2i(x, y)
			var rect := Rect2(origin + Vector2(x * CELL, y * CELL), Vector2(CELL - 2, CELL - 2))
			if world.is_lava(cell):
				draw_rect(rect, LAVA)
			elif world.is_goal(cell):
				draw_rect(rect, GOAL)
			else:
				draw_rect(rect, FLOOR)

	# where it has already been, so a wandering run reads as wandering rather than as
	# a dot moving at random
	for i in cursor + 1:
		var cell: Vector2i = path[i]
		if world.is_goal(cell):
			continue
		draw_rect(Rect2(origin + Vector2(cell.x * CELL + 10, cell.y * CELL + 10),
			Vector2(CELL - 22, CELL - 22)), TRAIL)

	var at: Vector2i = path[cursor]
	draw_circle(origin + Vector2(at.x * CELL + CELL * 0.5 - 1, at.y * CELL + CELL * 0.5 - 1),
		CELL * 0.32, agent_color)

	draw_string(font, origin + Vector2(0, world.height * CELL + 18),
		"step %d of %d" % [cursor, path.size() - 1], HORIZONTAL_ALIGNMENT_LEFT, -1, 13, TEXT_DIM)

# Steps per episode over the whole training. It is the only part of the picture that
# shows the learning happening rather than its result.
func _draw_curve(font: Font) -> void:
	# two points at least: the polyline divides by size() - 1
	if lengths.size() < 2:
		return
	var origin := Vector2(MARGIN, TOP + world.height * CELL + 52)
	var full_width := world.width * CELL * 2 + GAP
	var longest := 1
	for value in lengths:
		longest = max(longest, int(value))

	draw_string(font, origin + Vector2(0, -10), "steps per episode, %d to %d" % [
		int(lengths[0]), int(lengths[lengths.size() - 1])],
		HORIZONTAL_ALIGNMENT_LEFT, -1, 13, TEXT_DIM)

	var points := PackedVector2Array()
	for i in lengths.size():
		points.append(origin + Vector2(
			float(i) / float(lengths.size() - 1) * full_width,
			CURVE_H - float(lengths[i]) / float(longest) * CURVE_H))
	draw_polyline(points, AGENT_TAUGHT, 1.5, true)
