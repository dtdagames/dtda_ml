extends SceneTree

# Headless check of the linear regression demo, run with
#   godot --headless --script res://demo/verify_turret.gd
# Not part of the test suite: it proves the demo is worth showing, not that the library is correct.

func _initialize():
	var world := DTDADemoTurret.new()
	var tools := DTDATools.new()
	var rng := RandomNumberGenerator.new()
	rng.seed = 4242
	var failures := 0

	var sample := world.shots(rng, 300)
	var plain := world.fit_model(sample, false)
	var curved := world.fit_model(sample, true)

	if plain == null or curved == null:
		print("FAIL  fit() refused the training rows")
		quit(1)
		return

	# 1. the fit explains the shots it saw, reported for both because the gap between them is what the demo is about
	var r2_plain := _r2(tools, world, sample, plain, false)
	var r2_curved := _r2(tools, world, sample, curved, true)
	print("R2  distance only        : %.4f" % r2_plain)
	print("R2  distance and square  : %.4f" % r2_curved)
	if r2_plain < 0.9:
		print("FAIL  the one-feature fit explains too little to be worth showing")
		failures += 1
	if r2_curved <= r2_plain:
		print("FAIL  the square adds nothing, the demo has no point to make")
		failures += 1

	# 2. it hits, on targets it never trained on: one set of targets handed to all three, comparing means drawn from different samples would mix the luck of the draw into the difference reported
	var test_targets := world.targets(rng, 400)
	var miss_plain: Dictionary = world.evaluate(plain, false, test_targets)
	var miss_curved: Dictionary = world.evaluate(curved, true, test_targets)
	var miss_none: Dictionary = world.evaluate_untrained(test_targets)
	var reach := world.range_for(world.angle_max())
	var nearest := world.range_for(world.angle_min())
	print("mean miss  no model      : %6.2f m   (worst %.1f)" % [miss_none["mean"], miss_none["worst"]])
	print("mean miss  distance only : %6.2f m   (worst %.1f)" % [miss_plain["mean"], miss_plain["worst"]])
	print("mean miss  and square    : %6.2f m   (worst %.1f)" % [miss_curved["mean"], miss_curved["worst"]])
	print("the %d targets span the turret's reach, %.0f m to %.0f m" % [test_targets.size(), nearest, reach])

	if miss_plain["mean"] >= miss_none["mean"] / 3.0:
		print("FAIL  the trained turret is not clearly better than an untrained one")
		failures += 1
	if miss_curved["mean"] >= miss_plain["mean"]:
		print("FAIL  the extra feature does not improve the aim")
		failures += 1
	# an absolute floor as well as a comparison: "better than nothing" would still be true of a turret missing by ten metres, which is not something to put on a page
	if miss_curved["mean"] > 0.02 * reach:
		print("FAIL  even the better model misses by more than 2% of its reach")
		failures += 1

	# 3. the weights survive a round trip, the use case the README leads with: fit somewhere, ship the file, load it in the game
	var path := "user://dtda_demo_turret.json"
	if not curved.save(path):
		print("FAIL  save() refused the fitted model")
		failures += 1
	else:
		var reloaded := DTDALinReg.new(0.05, 4000)
		if not reloaded.load(path):
			print("FAIL  load() refused the file it just wrote")
			failures += 1
		else:
			var before := world.aim(curved, 60.0, true)
			var after := world.aim(reloaded, 60.0, true)
			if abs(before - after) > 0.0001:
				print("FAIL  the reloaded model aims differently: %f against %f" % [after, before])
				failures += 1

	print("")
	print("%d check(s) failed" % failures)
	quit(1 if failures > 0 else 0)

func _r2(tools: DTDATools, world: DTDADemoTurret, sample: Dictionary, model: DTDALinReg, quadratic: bool) -> float:
	var X: Array = []
	for distance in sample["distances"]:
		X.append(world.features(distance, quadratic))
	return tools.r2_score(model.predict(X), sample["angles"])
