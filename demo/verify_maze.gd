extends SceneTree

# Headless check of the demo. Not part of the test suite: it proves the demo is worth
# showing, not that the library is correct.
#
#   godot --headless --script res://demo/verify_maze.gd
#
# It answers the only question the demo makes a claim about: does the agent actually
# get better, and is the arena solvable at all.

func _initialize():
	var world := DTDADemoMaze.new()
	var failures := 0

	# 1. the arena has a path at all. A demo whose goal is unreachable would still
	# "learn" something and show an agent dying forever, which reads as a broken model
	# rather than a broken level.
	var shortest := _bfs(world)
	if shortest < 0:
		print("FAIL  the goal is unreachable, the arena is not solvable")
		failures += 1
	else:
		print("shortest possible path : %d steps" % shortest)

	var result := world.train(400, 12345)
	var first: Dictionary = result["first"]
	var learned: Dictionary = result["learned"]

	print("episode   1 : %d steps, reward %.0f, reached %s" % [
		first["path"].size() - 1, first["reward"], first["reached"]])
	print("after 400   : %d steps, reward %.0f, reached %s" % [
		learned["path"].size() - 1, learned["reward"], learned["reached"]])

	# 2. the trained agent reaches the goal. This is the claim the demo makes visually.
	if not learned["reached"]:
		print("FAIL  the trained agent does not reach the goal")
		failures += 1

	# 3. it is not merely lucky: the learned path is at most two steps off the best
	# possible one. A path that merely survives would not look like learning.
	if shortest >= 0 and learned["path"].size() - 1 > shortest + 2:
		print("FAIL  the learned path is %d steps against %d possible" % [
			learned["path"].size() - 1, shortest])
		failures += 1

	# 4. the contrast is visible. The whole point of the demo is the before/after, so
	# a first episode that already looks competent makes it pointless. Luck is allowed
	# to carry an untrained agent to the goal; taking a comparable route is not.
	var first_steps: int = first["path"].size() - 1
	var learned_steps: int = learned["path"].size() - 1
	if first_steps < learned_steps * 3:
		print("FAIL  the first episode takes %d steps against %d learned, too close to show" % [
			first_steps, learned_steps])
		failures += 1

	# 5. the curve goes down. Averaged over blocks, because single episodes are noisy
	# while epsilon is still high.
	var lengths: Array = result["lengths"]
	# the block is capped by what was actually run: slice(size - 50) walks backwards
	# from the end and would take the whole array on a shorter training
	var block: int = min(50, int(lengths.size() / 2.0))
	var early := _mean(lengths.slice(0, block))
	var late := _mean(lengths.slice(lengths.size() - block))
	print("mean length : %.1f over the first %d episodes, %.1f over the last %d" % [
		early, block, late, block])
	if late >= early:
		print("FAIL  episodes do not get shorter")
		failures += 1

	print("")
	print("%d check(s) failed" % failures)
	quit(1 if failures > 0 else 0)

func _mean(values: Array) -> float:
	if values.is_empty():
		return 0.0
	var total := 0.0
	for v in values:
		total += float(v)
	return total / float(values.size())

# Breadth-first search over the walkable cells, to know the best possible path
# independently of anything the agent does.
func _bfs(world: DTDADemoMaze) -> int:
	var queue: Array[Vector2i] = [world.start_cell]
	var seen := {world.start_cell: 0}
	while not queue.is_empty():
		var cell: Vector2i = queue.pop_front()
		if world.is_goal(cell):
			return seen[cell]
		for action in DTDADemoMaze.ACTIONS:
			var next := world.moved(cell, action)
			if next == cell or seen.has(next) or world.is_lava(next):
				continue
			seen[next] = seen[cell] + 1
			queue.append(next)
	return -1
