extends RefCounted

class_name DTDADemoMaze

# The world the demo agent learns in, kept apart from anything that draws it so the
# learning can be verified headless. demo/verify_maze.gd does exactly that.
#
# An open arena rather than a corridor maze: the agent can move anywhere, and what
# it has to learn is where NOT to go. That is what makes the before/after visible —
# an untrained agent walks into the lava within a few steps, a trained one threads
# between the patches.

const GRID = [
	"...##......G",
	"...##.###...",
	".....#...##.",
	"..##.#.#....",
	"..##...#.##.",
	"....##.#....",
	".##..#...##.",
	"S...........",
]

const ACTIONS = ["up", "down", "left", "right"]

# One step of walking costs a little, so a wandering agent is worse than a direct one
# even when both survive. Without it every surviving path scores the same and the
# agent has no reason to shorten anything.
const STEP_REWARD = -1.0
# Lava hurts and blocks, it does not kill. Ending the episode on contact was the first
# design and it does not learn: the agent dies within about ten steps while the goal is
# eighteen away, so it never once collects the reward it is supposed to be chasing and
# has nothing to propagate. Verified rather than reasoned - demo/verify_maze.gd caught
# it, with a trained agent no better than an untrained one.
const LAVA_REWARD = -25.0
const GOAL_REWARD = 100.0
const MAX_STEPS = 200

var width: int
var height: int
var start_cell: Vector2i
var goal_cell: Vector2i

func _init() -> void:
	height = GRID.size()
	width = GRID[0].length()
	for y in height:
		for x in width:
			match GRID[y][x]:
				"S": start_cell = Vector2i(x, y)
				"G": goal_cell = Vector2i(x, y)

func is_lava(cell: Vector2i) -> bool:
	return GRID[cell.y][cell.x] == "#"

func is_goal(cell: Vector2i) -> bool:
	return cell == goal_cell

# The state handed to the agent is an integer, never a float. The q table keys on
# str(), and str(2.0) prints "2" up to Godot 4.3 and "2.0" from 4.4 on, so a float
# state would give a table that does not survive an engine upgrade.
func state_of(cell: Vector2i) -> int:
	return cell.y * width + cell.x

func moved(cell: Vector2i, action: String) -> Vector2i:
	var next := cell
	match action:
		"up": next.y -= 1
		"down": next.y += 1
		"left": next.x -= 1
		"right": next.x += 1
	# walking into the edge leaves it where it is, it does not end the episode
	if next.x < 0 or next.x >= width or next.y < 0 or next.y >= height:
		return cell
	return next

# Answers {cell, reward, done, burnt} rather than mutating anything, so the same call
# is usable by the training loop and by the replay that draws it.
func step(cell: Vector2i, action: String) -> Dictionary:
	var next := moved(cell, action)
	# Lava blocks and hurts: the agent stays on the cell it came from. Nothing ends the
	# episode but reaching the goal, or running out of steps.
	if is_lava(next):
		return {"cell": cell, "reward": LAVA_REWARD, "done": false, "burnt": true}
	if is_goal(next):
		return {"cell": next, "reward": GOAL_REWARD, "done": true, "burnt": false}
	return {"cell": next, "reward": STEP_REWARD, "done": false, "burnt": false}

# Plays one episode and returns the path walked, so the demo can replay a run rather
# than re-simulate it. greedy = true takes the learned policy with no exploration.
func play(agent: DTDAQLearning, learning: bool) -> Dictionary:
	var cell := start_cell
	var path: Array[Vector2i] = [cell]
	var total := 0.0
	var reached := false
	for i in MAX_STEPS:
		var action = agent.choose_action(state_of(cell), ACTIONS) if learning else agent.predict(state_of(cell), ACTIONS)
		var outcome := step(cell, action)
		if learning:
			agent.learn(state_of(cell), action, outcome["reward"],
				state_of(outcome["cell"]), ACTIONS, outcome["done"])
		cell = outcome["cell"]
		path.append(cell)
		total += outcome["reward"]
		if outcome["done"]:
			reached = is_goal(cell)
			break
	return {"path": path, "reward": total, "reached": reached}

# The whole training, returned rather than printed: the first episode, the last one,
# and the length of every episode in between for the curve the demo draws.
func train(episodes: int, seed_value: int) -> Dictionary:
	var agent := DTDAQLearning.new(0.2, 0.95, 1.0, 0.99, 0.02)
	agent.set_seed(seed_value)
	var lengths: Array[int] = []
	var first := {}
	for e in episodes:
		var run := play(agent, true)
		if e == 0:
			first = run
		lengths.append(run["path"].size() - 1)
		# epsilon does not decay on its own: DTDAQLearning leaves it to the caller,
		# because only the caller knows where an episode ends. Forgetting this call
		# leaves the agent exploring at full rate forever - it still learns a good
		# table, but it never uses it, and every episode looks as random as the first.
		agent.decay_exploration()
	return {
		"agent": agent,
		"first": first,
		"learned": play(agent, false),
		"lengths": lengths,
	}
