# DTDAQLearning, the tabular Q-learning agent.

# A four room corridor: 0 - 1 - 2 - 3.
# "right" moves one room up, "left" one room down, both blocked at the ends.
# Entering room 3 ends the episode and pays 1, every other move pays 0.
const GOAL = 3
const ACTIONS = ["left", "right"]
const GAMMA = 0.9

# the optimal q values of that corridor, computed by hand with gamma = 0.9:
#   V*(3) = 0, the episode is over
#   Q*(2, right) = 1                     the reward, no future
#   Q*(1, right) = 0.9 * V*(2) = 0.9     V*(2) = 1
#   Q*(0, right) = 0.9 * V*(1) = 0.81    V*(1) = 0.9
#   Q*(0, left)  = 0.9 * V*(0) = 0.729   staying in 0, V*(0) = 0.81
#   Q*(1, left)  = 0.9 * V*(0) = 0.729
#   Q*(2, left)  = 0.9 * V*(1) = 0.81
const OPTIMAL = {
	"0": {"left": 0.729, "right": 0.81},
	"1": {"left": 0.729, "right": 0.9},
	"2": {"left": 0.81, "right": 1.0},
}

# returns [next state, reward, done]
func _step(state, action):
	var next_state = clamp(state + (1 if action == "right" else -1), 0, GOAL)
	return [next_state, 1.0 if next_state == GOAL else 0.0, next_state == GOAL]

func _train_corridor(agent, episodes):
	for episode in episodes:
		var state = 0
		# a bounded episode: random moves could otherwise bounce forever
		for step in 50:
			var action = agent.choose_action(state, ACTIONS)
			var result = _step(state, action)
			agent.learn(state, action, result[1], result[0], ACTIONS, result[2])
			state = result[0]
			if result[2]:
				break
		agent.decay_exploration()

# write a handmade file and try to load it, for the malformed file guards
func _load_written(content, agent = null):
	var path = "user://dtda_ml_test_qlearning_handmade.json"
	var file = FileAccess.open(path, FileAccess.WRITE)
	file.store_string(content)
	file.close()
	if agent == null:
		agent = DTDAQLearning.new()
	return agent.load(path)

# how many assertions this suite runs, checked by the runner
const PLAN = 89

func _run(t):
	t.section("Q-Learning, the Bellman update step by step")
	# every number below is computed by hand, not read off the model
	var solo = DTDAQLearning.new(0.5, 0.9, 0.0)
	# first update on an empty table: 0 + 0.5 * (1 + 0.9 * 0 - 0)
	solo.learn("a", "go", 1, "b", ["go"], false)
	t.check_near("an unknown next state carries no future value", solo.get_q("a", "go"), 0.5)
	# "b" is terminal here: 0 + 0.5 * 2
	solo.learn("b", "go", 2, "end", [], true)
	t.check_near("a terminal transition is worth lr * reward", solo.get_q("b", "go"), 1.0)
	# now "b" is worth 1.0: 0.5 + 0.5 * (1 + 0.9 * 1.0 - 0.5)
	solo.learn("a", "go", 1, "b", ["go"], false)
	t.check_near("the value of the next state flows back", solo.get_q("a", "go"), 1.2)
	# the arithmetic of the update, with an integer reward mixed into it:
	# 1.2 + 0.5 * (3 + 0.9 * 1.0 - 1.2)
	solo.learn("a", "go", 3, "b", ["go"], false)
	t.check_near("an integer reward mixes into the update without truncating", solo.get_q("a", "go"), 2.55)

	t.section("Q-Learning, bootstrapping without a list of next actions")
	# next_actions omitted, the documented fallback: whatever is already known there
	var boot = DTDAQLearning.new(1.0, 0.9, 0.0)
	boot.learn("b", "go", 10, "end", [], true)
	boot.learn("a", "go", 0, "b")
	t.check_near("an omitted next action list still looks at the next state",
		boot.get_q("a", "go"), 9.0)
	# and a null one is read the same way, like _choose_action already does
	var nulled = DTDAQLearning.new(1.0, 0.9, 0.0)
	nulled.learn("b", "go", 10, "end", [], true)
	nulled.learn("a", "go", 0, "b", null, false)
	t.check_near("a null next action list behaves like an omitted one",
		nulled.get_q("a", "go"), 9.0)
	t.check_equal("_predict takes a null action list too", nulled.predict("b", null), "go")
	# the fallback reads the row of the next state, not the whole table: a fortune
	# learned in "c" must not raise what "a" expects from "b"
	boot.learn("c", "wait", 100, "end", [], true)
	boot.learn("a", "go", 0, "b")
	t.check_near("the fallback stays inside the next state", boot.get_q("a", "go"), 9.0)

	t.section("Q-Learning, a reward that is not a number")
	# a reward arrives from whatever the game computed, and it lands straight in the
	# table. A nan there answers false to every comparison, so _best_action() can no
	# longer name a best action for that state and predict() falls silent
	var zero = 0.0
	var poisoned = DTDAQLearning.new(0.5, 0.9, 0.0)
	poisoned.learn("room", "north", 10, "end", [], true)
	var kept = poisoned.get_q("room", "north")
	t.check("a nan reward is refused", poisoned.learn("room", "north", zero / zero, "end", [], true) == null)
	t.check("an infinite reward is refused too",
		poisoned.learn("room", "north", 1.0 / zero, "end", [], true) == null)
	t.check("a reward that is not a number at all is refused",
		poisoned.learn("room", "north", "nope", "end", [], true) == null)
	t.check_near("the cell keeps the value it had", poisoned.get_q("room", "north"), kept, 0.0)
	t.check_equal("and the state can still name its best action", poisoned.predict("room"), "north")

	t.section("Q-Learning, terminal transitions")
	var term = DTDAQLearning.new(1.0, 0.9, 0.0)
	term.learn("rich", "x", 10, "end", [], true)
	t.check_near("a full learning rate takes the target as is", term.get_q("rich", "x"), 10.0)
	# 1 + 0.9 * 10 when the transition continues
	term.learn("s", "go", 1, "rich", ["x"], false)
	t.check_near("a normal transition adds the discounted future", term.get_q("s", "go"), 10.0)
	# the very same transition marked terminal drops the 9.0 of future
	term.learn("s", "stop", 1, "rich", ["x"], true)
	t.check_near("a terminal transition ignores the next state", term.get_q("s", "stop"), 1.0)

	t.section("Q-Learning, convergence on the corridor")
	var agent = DTDAQLearning.new(0.2, GAMMA, 1.0, 0.999, 0.1)
	agent.set_seed(20240817)
	_train_corridor(agent, 3000)
	for state in OPTIMAL:
		for action in OPTIMAL[state]:
			t.check_near("Q(%s, %s) reaches its optimal value" % [state, action],
				agent.get_q(int(state), action), OPTIMAL[state][action], 0.01)
	t.check_equal("the learned policy walks to the goal",
		[agent.predict(0), agent.predict(1), agent.predict(2)], ["right", "right", "right"])
	# nothing was ever learned about the goal, the episode ends there
	t.check_near("the terminal state stays empty", agent.get_q(GOAL, "right"), 0.0)

	t.section("Q-Learning, exploitation and determinism")
	agent.exploration_rate = 0.0
	var picks = []
	for i in 20:
		picks.push_back(agent.choose_action(0, ACTIONS))
	t.check_equal("epsilon 0 gives the same action every time", picks.count("right"), 20)
	t.check_equal("_predict agrees with a greedy _choose_action",
		agent.predict(0, ACTIONS), agent.choose_action(0, ACTIONS))
	# a state nobody ever visited: every action is worth 0, the first of the list wins
	var fresh = DTDAQLearning.new(0.1, 0.9, 0.0)
	fresh.learn("elsewhere", "wait", 0, "elsewhere", [], true)
	t.check_equal("a tie keeps the first action of the list", fresh.choose_action("void", ACTIONS), "left")
	t.check_equal("and follows the order it was given", fresh.choose_action("void", ["right", "left"]), "right")

	t.section("Q-Learning, exploration")
	var explorer = DTDAQLearning.new(0.1, 0.9, 1.0)
	explorer.set_seed(7)
	explorer.learn(0, "right", 1, 1, ACTIONS, true)
	var seen = {}
	for i in 200:
		seen[explorer.choose_action(0, ACTIONS)] = true
	# epsilon 1.0 ignores the q values entirely, both actions must show up
	t.check_equal("epsilon 1 draws every action", seen.size(), 2)

	t.section("Q-Learning, exploration decay")
	var eps = DTDAQLearning.new(0.1, 0.9, 1.0, 0.5, 0.2)
	t.check_near("epsilon starts where it was set", eps.exploration_rate, 1.0)
	t.check_near("_decay_exploration returns the new rate", eps.decay_exploration(), 0.5)
	eps.decay_exploration()
	t.check_near("it decays once per episode", eps.exploration_rate, 0.25)
	# 0.25 * 0.5 = 0.125, below the floor
	eps.decay_exploration()
	t.check_near("epsilon never goes below its floor", eps.exploration_rate, 0.2)
	eps.decay_exploration()
	t.check_near("and stays on the floor", eps.exploration_rate, 0.2)
	# epsilon is a probability, a decay above 1 must not push it past "always explore"
	var rising = DTDAQLearning.new(0.1, 0.9, 0.5, 2.0, 0.01)
	rising.decay_exploration()
	rising.decay_exploration()
	rising.decay_exploration()
	t.check_near("epsilon never goes above 1", rising.exploration_rate, 1.0)
	# the same interval holds for what is asked at build time, floor included:
	# a floor above 1 would otherwise contradict the range the class promises
	var absurd = DTDAQLearning.new(0.1, 0.9, 5.0, 0.5, 1.5)
	t.check_near("an exploration rate above 1 is brought back", absurd.exploration_rate, 1.0)
	t.check_near("so is a floor above 1", absurd.min_exploration_rate, 1.0)
	absurd.decay_exploration()
	t.check_near("and the decay keeps it there", absurd.exploration_rate, 1.0)

	t.section("Q-Learning, reset")
	eps.learn("s", "a", 1, "end", [], true)
	eps.reset()
	t.check_near("_reset puts the exploration back", eps.exploration_rate, 1.0)
	t.check("_reset forgets the table", eps.q_table == null)
	# a seeded agent must replay the very same run after a reset, otherwise the
	# reproducibility set_seed() promises only holds until the first reset
	var replay = DTDAQLearning.new(0.1, 0.9, 1.0)
	replay.set_seed(99)
	var first_run = []
	for i in 12:
		first_run.push_back(replay.choose_action("s", ACTIONS))
	replay.reset()
	var second_run = []
	for i in 12:
		second_run.push_back(replay.choose_action("s", ACTIONS))
	t.check_equal("_reset replays the same random draws", second_run, first_run)

	t.section("Q-Learning, arbitrary states and actions")
	# states are not contiguous integers here, and never touch each other
	var grid = DTDAQLearning.new(1.0, 0.9, 0.0)
	grid.learn(Vector2i(3, -7), "north", 5, Vector2i(3, -6), [], true)
	grid.learn(Vector2i(3, -7), "south", 1, Vector2i(3, -8), [], true)
	t.check_equal("a Vector2i state keeps its own row", grid.predict(Vector2i(3, -7)), "north")
	t.check_near("and its own values", grid.get_q(Vector2i(3, -7), "south"), 1.0)
	t.check_near("an unvisited state is worth 0", grid.get_q(Vector2i(0, 0), "north"), 0.0)

	# the price of keying by str(), pinned here so it stays a documented limit and not
	# a surprise: two actions with the same str() are one and the same cell, and the
	# type registry keeps the last one for the whole agent.
	# the pair is the integer 2 and the string "2", not 2 and 2.0: str() of a whole
	# float prints "2" up to Godot 4.3 and "2.0" from 4.4 on, so a float pair would
	# pin the number formatting of the engine instead of the rule of the model.
	# int and String have kept the same str() all along, and 2 next to "2" is the
	# realistic mistake anyway, an action read from a config file next to one written
	# in code
	var collide = DTDAQLearning.new(1.0, 0.9, 0.0)
	collide.learn("s", 2, 1, "end", [], true)
	collide.learn("s", "2", 5, "end", [], true)
	# through _known_actions() rather than q_table["s"], so the assertion asks how many
	# cells that row holds without spelling the internal key out itself
	t.check_equal("two actions with the same str() share one cell", collide._known_actions("s").size(), 1)
	t.check_near("the second one overwrites the first", collide.get_q("s", 2), 5.0)
	# this one is about the type registry alone, not about the collision: "2" was
	# learned second and carries the higher q value, so it would come out on top of
	# two separate cells just as well. The collision is what the two lines above pin
	t.check_equal("and the last type learned wins", typeof(collide.predict("s")), TYPE_STRING)
	# across two states there is no shared cell, the values stay apart...
	var apart = DTDAQLearning.new(1.0, 0.9, 0.0)
	apart.learn("roomA", 2, 1, "end", [], true)
	apart.learn("roomB", "2", 5, "end", [], true)
	t.check_equal("two states keep their own row for the same action key",
		[apart.get_q("roomA", 2), apart.get_q("roomB", "2")], [1.0, 5.0])
	# ...but the type registry is global: roomA played the integer 2 and is answered
	# the string "2", no cell being shared. That one bites, "2" == 2 raises in GDScript
	t.check_equal("the type of an action is global to the agent",
		typeof(apart.predict("roomA")), TYPE_STRING)

	t.section("Q-Learning, saving and loading")
	var path = "user://dtda_ml_test_qlearning.json"
	t.check("_save reports a success", agent.save(path))
	var back = DTDAQLearning.new()
	t.check("_load reports a success", back.load(path))
	# 1e-12 is far tighter than anything the model could get wrong, and still leaves room
	# for the last digit the JSON writer drops
	for state in OPTIMAL:
		for action in ACTIONS:
			t.check_near("Q(%s, %s) comes back untouched" % [state, action],
				back.get_q(int(state), action), agent.get_q(int(state), action), 1e-12)
	t.check_equal("a reloaded agent follows the same policy",
		[back.predict(0), back.predict(1), back.predict(2)],
		[agent.predict(0), agent.predict(1), agent.predict(2)])
	t.check_near("the hyperparameters come back too", back.discount_factor, GAMMA)
	t.check_near("including the exploration rate", back.exploration_rate, agent.exploration_rate)

	# the trap: a JSON key is always a string and JSON numbers always come back as floats
	var numeric_path = "user://dtda_ml_test_qlearning_int.json"
	var numeric = DTDAQLearning.new(1.0, 0.9, 0.0)
	numeric.learn("hall", 2, 1, "end", [], true)
	numeric.learn("hall", 7, 0, "end", [], true)
	t.check("an agent with integer actions saves", numeric.save(numeric_path))
	var numeric_back = DTDAQLearning.new()
	t.check("it loads", numeric_back.load(numeric_path))
	var picked = numeric_back.predict("hall")
	t.check_equal("an integer action comes back with its value", picked, 2)
	t.check("an integer action comes back as an int, not a string or a float",
		typeof(picked) == TYPE_INT)
	# a float action must not be rounded into an int on the way back
	var mixed_path = "user://dtda_ml_test_qlearning_float.json"
	var mixed = DTDAQLearning.new(1.0, 0.9, 0.0)
	mixed.learn("hall", 0.5, 3, "end", [], true)
	mixed.save(mixed_path)
	var mixed_back = DTDAQLearning.new()
	mixed_back.load(mixed_path)
	t.check_equal("a float action keeps its type", typeof(mixed_back.predict("hall")), TYPE_FLOAT)
	t.check_near("and its value", mixed_back.predict("hall"), 0.5)

	# a StringName is a type of its own and must come back as one
	var named_path = "user://dtda_ml_test_qlearning_name.json"
	var named = DTDAQLearning.new(1.0, 0.9, 0.0)
	named.learn("hall", &"jump", 1, "end", [], true)
	named.save(named_path)
	var named_back = DTDAQLearning.new()
	named_back.load(named_path)
	t.check_equal("a StringName action keeps its type",
		typeof(named_back.predict("hall")), TYPE_STRING_NAME)

	# a float key goes through str(), which may or may not carry every digit of a
	# double depending on the engine. What the model owes is that the file adds
	# nothing to that: the expectation is float(str(x)), not x, so the assertion
	# stays out of the number formatting business where 4.3 and 4.4 differ. Both
	# sides move together when str() changes, which is what makes it version proof.
	# it is not str() compared with itself either: the left hand side went to disk
	# and back through JSON, and a _key() rounding more than str() still fails here
	var third_path = "user://dtda_ml_test_qlearning_third.json"
	var third = DTDAQLearning.new(1.0, 0.9, 0.0)
	third.learn("hall", 1.0 / 3.0, 1, "end", [], true)
	third.save(third_path)
	var third_back = DTDAQLearning.new()
	third_back.load(third_path)
	t.check_equal("a long float action still comes back as a float",
		typeof(third_back.predict("hall")), TYPE_FLOAT)
	# no tolerance at all: the round trip through the file must be exact
	t.check_near("a float key survives the file as well as str() allows",
		third_back.predict("hall"), float(str(1.0 / 3.0)), 0.0)

	# the type is written as a stable label, not as the raw value of an engine enum
	var raw_file = FileAccess.open(numeric_path, FileAccess.READ)
	var raw = JSON.parse_string(raw_file.get_as_text())
	raw_file.close()
	t.check_equal("the action type is written as a stable label", raw["actions"]["2"], "int")
	# the format of that field changed with version 2, the field must say so
	t.check_equal("the file announces its format version", int(raw["version"]), 2)
	# a version 1 file holds raw enum values under "actions" and cannot be told apart
	# from a version 2 one by its content: it must be refused, not read as strings
	t.check_equal("_load refuses the format of version 1",
		_load_written('{"model": "DTDAQLearning", "version": 1, "q_table": {"s": {"2": 1.0}}, "actions": {"2": 2}}'),
		false)
	t.check_equal("_load refuses a file with no version at all",
		_load_written('{"model": "DTDAQLearning", "q_table": {"s": {"a": 1.0}}}'), false)

	# to_dict() is public, what it hands out must not be the table the agent keeps using
	var snapshot = agent.to_dict()
	snapshot["q_table"]["0"]["right"] = 999.0
	t.check_near("_to_dict answers a copy of the table", agent.get_q(0, "right"), 0.81, 0.01)

	t.section("Q-Learning guards (the errors below are expected)")
	t.check("_predict before any transition", DTDAQLearning.new().predict("s") == null)
	t.check_equal("_save before any transition fails", DTDAQLearning.new().save(path), false)
	t.check("_choose_action without any valid action", agent.choose_action(0, []) == null)
	t.check("_choose_action with a null action list", agent.choose_action(0, null) == null)
	t.check("_predict on a state the agent never met", agent.predict("nowhere") == null)
	# same answer with a list of actions: an unknown state is unknown either way, the
	# agent must not dress up a tie between zeros as a learned policy
	t.check("_predict on an unknown state, actions given", agent.predict("nowhere", ACTIONS) == null)
	# a state whose row was emptied by hand: the row is there, and it holds nothing
	var hollow = DTDAQLearning.new()
	t.check_equal("a state with an empty row still loads",
		_load_written('{"model": "DTDAQLearning", "version": 2, "q_table": {"s": {}}}', hollow), true)
	t.check("_predict on a state whose row is empty", hollow.predict("s") == null)
	# a file written by another model, saved here so this suite stays self contained
	var other = DTDAKNN.new(1)
	other.fit([[0]], [1])
	var other_path = "user://dtda_ml_test_not_a_qlearning.json"
	other.save(other_path)
	t.check_equal("_load refuses another kind of model", DTDAQLearning.new().load(other_path), false)
	t.check_equal("_load refuses a missing file", DTDAQLearning.new().load("user://no_such_agent.json"), false)
	# a file an agent could read from end to end, wrong on the "model" field alone:
	# the DTDAKNN file above is turned away by the guards on the structure
	t.check_equal("DTDAQLearning refuses a file that only lies about its model name",
		_load_written('{"model": "NotQLearning", "version": 2, "q_table": {"s": {"a": 1.0}}, "actions": {"a": "string"}}'), false)

	# a model file lives in user://, where a player can edit it by hand.
	# the answer must be exactly false, the one a guard returns: a script error inside
	# _from_dict would leave _load with null, which "not" would happily accept
	# version 2, deliberately: _from_dict checks the version before the structure, so a
	# version 1 file would leave at the version guard and never reach the guard named here
	t.check_equal("_load refuses a q table that is not a table",
		_load_written('{"model": "DTDAQLearning", "version": 2, "q_table": [1, 2]}'), false)
	t.check_equal("_load refuses a state that holds no action",
		_load_written('{"model": "DTDAQLearning", "version": 2, "q_table": {"s": "nope"}}'), false)
	t.check_equal("_load refuses a q value that is not a number",
		_load_written('{"model": "DTDAQLearning", "version": 2, "q_table": {"s": {"a": "nope"}}}'), false)
	# a broken file must not wipe an agent that was already working
	var survivor = DTDAQLearning.new(1.0, 0.9, 0.0)
	survivor.learn("s", "a", 4, "end", [], true)
	var broken_path = "user://dtda_ml_test_qlearning_broken.json"
	var broken_file = FileAccess.open(broken_path, FileAccess.WRITE)
	broken_file.store_string('{"model": "DTDAQLearning", "version": 2, "q_table": {"s": "nope"}}')
	broken_file.close()
	survivor.load(broken_path)
	t.check_near("a refused file leaves the agent alone", survivor.get_q("s", "a"), 4.0)
