extends Node2D

# The linear regression demo: a turret that never learns the ballistic formula, only
# the shots it took.
#
# Left, the three hundred rounds it fired at random angles and the two fits made from
# them. Right, the same target engaged by both models at once. The difference between
# the two shells is the whole argument: one feature, then the same feature squared.
#
# As in the Q-Learning demo, the fitting happens once in _ready(); what runs per frame
# is only the arc of two shells.

const MARGIN := 28
const TOP := 132
const PANEL_W := 400
const PANEL_H := 230
const GAP := 44
const SAMPLES := 300
const SEED := 4242
const FLIGHT_SECONDS := 1.15
const PAUSE_SECONDS := 0.85

const BACKGROUND := Color("14161c")
const PANEL := Color("1b1f29")
const GROUND := Color("343b4a")
const DOT := Color("707d96")
const PLAIN := Color("d9a441")
const CURVED := Color("4a90d9")
const TARGET := Color("46a758")
const TEXT := Color("c8cdd8")
const TEXT_DIM := Color("6b7385")

var world: DTDADemoTurret
var sample: Dictionary
var plain: DTDALinReg
var curved: DTDALinReg
var rng := RandomNumberGenerator.new()

var reach := 1.0
var target_distance := 0.0
var angle_plain := 0.0
var angle_curved := 0.0
var shot_elapsed := 0.0
var rounds := 0
var total_plain := 0.0
var total_curved := 0.0
var total_untrained := 0.0
var seen_low := 0.0
var seen_high := 0.0

func _ready() -> void:
	world = DTDADemoTurret.new()
	rng.seed = SEED
	sample = world.shots(rng, SAMPLES)
	plain = world.fit_model(sample, false)
	curved = world.fit_model(sample, true)
	reach = world.range_for(world.angle_max())
	# the span the turret actually fired over. Anything outside it is extrapolation, and
	# drawing a fit there would show the model asserting things it never saw.
	seen_low = sample["distances"].min()
	seen_high = sample["distances"].max()
	_next_target()

func _next_target() -> void:
	target_distance = world.range_for(rng.randf_range(world.angle_min(), world.angle_max()))
	angle_plain = world.aim(plain, target_distance, false)
	angle_curved = world.aim(curved, target_distance, true)
	shot_elapsed = 0.0
	rounds += 1
	total_plain += abs(world.range_for(angle_plain) - target_distance)
	total_curved += abs(world.range_for(angle_curved) - target_distance)
	# the untrained turret is scored on the very same target, round by round, rather than
	# on a separate draw: three bars from three different samples would not compare
	var fixed := (world.angle_min() + world.angle_max()) * 0.5
	total_untrained += abs(world.range_for(fixed) - target_distance)

func _process(delta: float) -> void:
	shot_elapsed += delta
	if shot_elapsed > FLIGHT_SECONDS + PAUSE_SECONDS:
		_next_target()
	queue_redraw()

func _draw() -> void:
	if world == null or plain == null or curved == null:
		return
	var font := ThemeDB.fallback_font
	draw_rect(get_viewport_rect(), BACKGROUND)

	draw_string(font, Vector2(MARGIN, 44), "A turret that learns to aim by missing",
		HORIZONTAL_ALIGNMENT_LEFT, -1, 24, TEXT)
	draw_string(font, Vector2(MARGIN, 68),
		"It is never given the ballistic formula. It fires %d rounds at random angles and fits what it saw." % SAMPLES,
		HORIZONTAL_ALIGNMENT_LEFT, -1, 15, TEXT_DIM)
	draw_string(font, Vector2(MARGIN, 88),
		"Both shells come from one DTDALinReg. The blue one is also given the square of the distance.",
		HORIZONTAL_ALIGNMENT_LEFT, -1, 15, TEXT_DIM)

	_draw_sample(font)
	_draw_range(font)
	_draw_scores(font)

# The training set as it is: a distance measured on the ground against the angle that
# produced it, with the two fits laid over it.
func _draw_sample(font: Font) -> void:
	var origin := Vector2(MARGIN, TOP)
	draw_string(font, origin + Vector2(0, -12), "What it saw, and what it fitted",
		HORIZONTAL_ALIGNMENT_LEFT, -1, 16, TEXT)
	draw_rect(Rect2(origin, Vector2(PANEL_W, PANEL_H)), PANEL)

	# the axis is opened two degrees past the band on each side: a fit is free to predict
	# slightly outside what the turret can fire, and clipping that off would hide it
	var pad := deg_to_rad(2.0)
	var lo := world.angle_min() - pad
	var hi := world.angle_max() + pad
	var line_plain := PackedVector2Array()
	var line_curved := PackedVector2Array()
	for step in 61:
		var d: float = seen_low + (seen_high - seen_low) * float(step) / 60.0
		line_plain.append(origin + Vector2(d / reach * PANEL_W,
			PANEL_H - (world.raw_angle(plain, d, false) - lo) / (hi - lo) * PANEL_H))
		line_curved.append(origin + Vector2(d / reach * PANEL_W,
			PANEL_H - (world.raw_angle(curved, d, true) - lo) / (hi - lo) * PANEL_H))
	draw_polyline(line_plain, PLAIN, 2.0, true)
	draw_polyline(line_curved, CURVED, 2.0, true)

	# the shots last: a good fit sits on top of them, and a cloud drawn first would be
	# hidden by the very lines it is there to justify
	for i in sample["distances"].size():
		var d: float = sample["distances"][i]
		var a: float = sample["angles"][i]
		draw_circle(origin + Vector2(d / reach * PANEL_W, PANEL_H - (a - lo) / (hi - lo) * PANEL_H), 2.0, DOT)

	draw_string(font, origin + Vector2(0, PANEL_H + 16),
		"distance on the ground, %d to %d m, the span it fired over" % [int(seen_low), int(seen_high)], HORIZONTAL_ALIGNMENT_LEFT, -1, 13, TEXT_DIM)
	draw_string(font, origin + Vector2(0, PANEL_H + 32),
		"angle, %d to %d degrees, the band it may fire in" % [int(DTDADemoTurret.ANGLE_MIN_DEG), int(DTDADemoTurret.ANGLE_MAX_DEG)],
		HORIZONTAL_ALIGNMENT_LEFT, -1, 13, TEXT_DIM)

# The shot itself, both shells at once against a target neither model trained on.
func _draw_range(font: Font) -> void:
	var origin := Vector2(MARGIN + PANEL_W + GAP, TOP)
	draw_string(font, origin + Vector2(0, -12), "Firing at a target it never trained on",
		HORIZONTAL_ALIGNMENT_LEFT, -1, 16, TEXT)
	draw_rect(Rect2(origin, Vector2(PANEL_W, PANEL_H)), PANEL)

	var ground := origin + Vector2(0, PANEL_H - 26)
	var scale_x := PANEL_W / (reach * 1.06)
	draw_line(ground, ground + Vector2(PANEL_W, 0), GROUND, 2.0)

	var tx := ground + Vector2(target_distance * scale_x, 0)
	draw_rect(Rect2(tx + Vector2(-3, -14), Vector2(6, 14)), TARGET)

	# min() answers a Variant, so the type is written out rather than inferred
	var t: float = min(shot_elapsed, FLIGHT_SECONDS) / FLIGHT_SECONDS
	_draw_shell(ground, scale_x, angle_plain, t, PLAIN)
	_draw_shell(ground, scale_x, angle_curved, t, CURVED)

	# the turret last, so it sits over the muzzle of both arcs
	draw_rect(Rect2(ground + Vector2(-5, -11), Vector2(11, 11)), GROUND)

	if shot_elapsed >= FLIGHT_SECONDS:
		var mp: float = world.range_for(angle_plain) - target_distance
		var mc: float = world.range_for(angle_curved) - target_distance
		draw_string(font, origin + Vector2(0, PANEL_H + 16),
			"distance only : %+.1f m" % mp, HORIZONTAL_ALIGNMENT_LEFT, -1, 13, PLAIN)
		draw_string(font, origin + Vector2(0, PANEL_H + 32),
			"and its square : %+.1f m" % mc, HORIZONTAL_ALIGNMENT_LEFT, -1, 13, CURVED)

func _draw_shell(ground: Vector2, scale_x: float, angle: float, t: float, color: Color) -> void:
	var flight := world.flight_time(angle)
	var arc := PackedVector2Array()
	for step in 41:
		var time := flight * t * float(step) / 40.0
		var p := world.point_at(angle, time)
		arc.append(ground + Vector2(p.x * scale_x, -p.y * scale_x))
	if arc.size() > 1:
		draw_polyline(arc, Color(color, 0.45), 1.5, true)
		draw_circle(arc[arc.size() - 1], 3.0, color)

# The three numbers side by side, as bars: what it costs to fire without a model, with
# one feature, and with two.
func _draw_scores(font: Font) -> void:
	var origin := Vector2(MARGIN, TOP + PANEL_H + 66)
	var width := PANEL_W * 2 + GAP
	var rows := [
		["no model, fires at the middle of its band", total_untrained / float(max(rounds, 1)), GROUND],
		["distance only", total_plain / float(max(rounds, 1)), PLAIN],
		["distance and its square", total_curved / float(max(rounds, 1)), CURVED],
	]
	draw_string(font, origin + Vector2(0, -12), "average miss over %d targets, the turret reaches %d m" % [
		rounds, int(reach)], HORIZONTAL_ALIGNMENT_LEFT, -1, 14, TEXT_DIM)
	var longest: float = max(total_untrained / float(max(rounds, 1)), 0.001)
	for i in rows.size():
		var y: float = origin.y + i * 26
		var value: float = rows[i][1]
		draw_rect(Rect2(Vector2(origin.x, y), Vector2(value / longest * (width - 300), 14)), rows[i][2])
		draw_string(font, Vector2(origin.x + width - 290, y + 12), "%5.2f m   %s" % [value, rows[i][0]],
			HORIZONTAL_ALIGNMENT_LEFT, -1, 13, TEXT_DIM)
