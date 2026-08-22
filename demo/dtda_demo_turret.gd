extends RefCounted

class_name DTDADemoTurret

# The world of the linear regression demo, kept apart from anything that draws it so
# the fit can be checked headless. demo/verify_turret.gd does that.
#
# A turret fires at a fixed muzzle speed and only chooses its angle. It is never told
# the ballistic formula: it fires a few hundred times at random angles, watches where
# the shells land, and fits the angle needed for a distance from what it saw.
#
# The point the demo makes is not that regression is magic here - the true relation is
# an arcsine and no straight line matches it exactly. It is that the same model does
# noticeably better once you hand it the square of the distance as a second feature.
# Linear regression is linear in the FEATURES you give it, not in the world.

const SPEED := 30.0
const GRAVITY := 9.8
# The band the turret is allowed to use. Near 45 degrees the range stops responding to
# the angle, which no fit can help with, so the demo stays below it.
const ANGLE_MIN_DEG := 10.0
const ANGLE_MAX_DEG := 40.0

func angle_min() -> float:
	return deg_to_rad(ANGLE_MIN_DEG)

func angle_max() -> float:
	return deg_to_rad(ANGLE_MAX_DEG)

# Where a shell fired at this angle lands, on flat ground.
func range_for(angle: float) -> float:
	return SPEED * SPEED * sin(2.0 * angle) / GRAVITY

func flight_time(angle: float) -> float:
	return 2.0 * SPEED * sin(angle) / GRAVITY

# For drawing the arc. x along the ground, y up.
func point_at(angle: float, t: float) -> Vector2:
	return Vector2(SPEED * cos(angle) * t, SPEED * sin(angle) * t - 0.5 * GRAVITY * t * t)

# The training set: fire at random angles and record where it landed. This is the whole
# input - a distance measured on the ground and the angle that produced it.
func shots(rng: RandomNumberGenerator, count: int) -> Dictionary:
	var distances: Array = []
	var angles: Array = []
	for i in count:
		var angle := rng.randf_range(angle_min(), angle_max())
		distances.append(range_for(angle))
		angles.append(angle)
	return {"distances": distances, "angles": angles}

# One row of features for a distance. With quadratic, the square of the distance comes
# along: still one linear regression, fitting a curve because the feature is curved.
func features(distance: float, quadratic: bool) -> Array:
	return [distance, distance * distance] if quadratic else [distance]

func fit_model(sample: Dictionary, quadratic: bool) -> DTDALinReg:
	var X: Array = []
	for distance in sample["distances"]:
		X.append(features(distance, quadratic))
	var model := DTDALinReg.new(0.05, 4000)
	# fit() answers false when it refuses the rows; a demo that ignored that would draw
	# an untrained model and blame the library
	if not model.fit(X, sample["angles"]):
		return null
	return model

# What the model says, untouched. Drawn as it is by the demo: clamping first would bend
# the one-feature fit into something that looks curved, which is the very distinction
# the picture is there to make.
func raw_angle(model: DTDALinReg, target: float, quadratic: bool) -> float:
	var predicted: Array = model.predict([features(target, quadratic)])
	if predicted.is_empty():
		return angle_min()
	return float(predicted[0])

# The angle the turret will actually use, held inside the band it is allowed to fire in.
# Without the clamp a prediction just outside the range trained on turns into an angle
# the turret cannot take.
func aim(model: DTDALinReg, target: float, quadratic: bool) -> float:
	return clamp(raw_angle(model, target, quadratic), angle_min(), angle_max())

# A set of targets to be judged on. Drawn once and handed to every model, because a
# fresh draw per model would compare them on different shots: the means would differ by
# the luck of the sample as much as by the fit, and the worst case most of all.
func targets(rng: RandomNumberGenerator, count: int) -> Array:
	var picked: Array = []
	for i in count:
		picked.append(range_for(rng.randf_range(angle_min(), angle_max())))
	return picked

# How far off it lands, over targets it never trained on.
func evaluate(model: DTDALinReg, quadratic: bool, on_targets: Array) -> Dictionary:
	var total := 0.0
	var worst := 0.0
	for target in on_targets:
		var miss: float = abs(range_for(aim(model, target, quadratic)) - target)
		total += miss
		worst = max(worst, miss)
	return {"mean": total / float(max(on_targets.size(), 1)), "worst": worst}

# What a turret with no model does: always the middle of its band. The comparison the
# demo needs, and the honest one - a turret firing at random would look worse than
# anything and prove nothing. Same targets as the models, for the same reason.
func evaluate_untrained(on_targets: Array) -> Dictionary:
	var fixed := (angle_min() + angle_max()) * 0.5
	var total := 0.0
	var worst := 0.0
	for target in on_targets:
		var miss: float = abs(range_for(fixed) - target)
		total += miss
		worst = max(worst, miss)
	return {"mean": total / float(max(on_targets.size(), 1)), "worst": worst}
