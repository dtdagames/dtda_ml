# DTDAQLearning, the tabular Q-learning agent, on a four room corridor 0 - 1 - 2 - 3: "right" moves one room up, "left" one down, both blocked at the ends, entering room 3 ends the episode and pays 1, every other move pays 0. The values below are computed by hand with gamma 0.9, not read off the model: V*(3) = 0 the episode being over, Q*(2, right) = 1 the reward with no future, Q*(1, right) = 0.9 * V*(2) = 0.9, Q*(0, right) = 0.9 * V*(1) = 0.81, Q*(0, left) = Q*(1, left) = 0.9 * V*(0) = 0.729, Q*(2, left) = 0.9 * V*(1) = 0.81
const ACTIONS = ["left", "right"]
const OPTIMAL = [[0, "left", 0.729], [0, "right", 0.81], [1, "left", 0.729], [1, "right", 0.9], [2, "left", 0.81], [2, "right", 1.0]]
# one action of every scalar type the file rebuilds, 1.0 / 3.0 among them: a float key goes through str(), which may carry fewer digits than the double, so both sides of the assertion go through str() and move together when the engine changes how it prints a float
const TYPED = [2, 1.0 / 3.0, true, &"jump", "stay"]
# a model file lives in user://, where a player can edit it by hand: another model's name, no version at all, the version 1 format (raw engine enum values under "actions", which no content can tell from a version 2 file), and a q value that is not a number
const BROKEN = ['{"model": "NotQLearning", "version": 2, "q_table": {"s": {"a": 1.0}}}', '{"model": "DTDAQLearning", "q_table": {"s": {"a": 1.0}}}', '{"model": "DTDAQLearning", "version": 1, "q_table": {"s": {"2": 1.0}}, "actions": {"2": 2}}', '{"model": "DTDAQLearning", "version": 2, "q_table": {"s": {"a": "nope"}}}']
const PLAN = 50

func _run(t):
	t.section("Q-Learning, the Bellman update by hand, then the corridor")
	var one = DTDAQLearning.new(0.5, 0.9, 0.0)
	one.learn("a", "go", 1, "b", [], false)
	t.check_near("an unknown next state carries no future value", one.get_q("a", "go"), 0.5)
	one.learn("b", "go", 2, "end", [], true)
	t.check_near("a terminal transition is worth lr * reward", one.get_q("b", "go"), 1.0)
	one.learn("a", "go", 1, "b", ["go"], false)
	t.check_near("the value of the next state flows back, discounted", one.get_q("a", "go"), 1.2)
	one.learn("c", "wait", 100, "end", [], true)
	one.learn("a", "go", 1, "b")
	t.check_near("an omitted action list reads the row of the next state, and that row alone", one.get_q("a", "go"), 1.55)
	t.check_equal("_predict reads a null action list like an omitted one", one.predict("b", null), "go")
	# a nan reward answers false to every comparison, which would leave predict() unable to name a best action for that state ever again. check_equal against null, never "not learn()": a call that raises answers null too
	for bad in [NAN, INF, "nope"]: t.check_equal("a reward that is not a number is refused", one.learn("b", "go", bad, "end", [], true), null)
	t.check_near("and the cell it aimed at keeps the value it had", one.get_q("b", "go"), 1.0, 0.0)

	var agent = DTDAQLearning.new(0.2, 0.9, 1.0, 0.999, 0.1)
	agent.set_seed(20240817)
	for episode in 3000:
		var state = 0
		# a bounded episode: random moves could otherwise bounce forever
		for step in 50:
			var action = agent.choose_action(state, ACTIONS)
			var moved = clamp(state + (1 if action == "right" else -1), 0, 3)
			agent.learn(state, action, int(moved == 3), moved, ACTIONS, moved == 3)
			state = moved
			if state == 3: break
		agent.decay_exploration()
	for row in OPTIMAL: t.check_near("Q(%d, %s) reaches the value computed by hand" % [row[0], row[1]], agent.get_q(row[0], row[1]), row[2], 0.01)
	t.check_equal("a trained agent walks to the goal, where a new one would know nothing", [agent.predict(0), agent.predict(1), agent.predict(2)], ["right", "right", "right"])
	agent.exploration_rate = 0.0
	var greedy = {}
	for i in 20: greedy[agent.choose_action(0, ACTIONS)] = true
	t.check_equal("epsilon 0 never explores, twenty draws take the best action known", [greedy.size(), agent.predict(0, ACTIONS)], [1, "right"])
	t.check_equal("a tie keeps the first action of the list it was given", [agent.choose_action("void", ACTIONS), agent.choose_action("void", ["right", "left"])], ["left", "right"])
	var eps = DTDAQLearning.new(0.1, 0.9, 1.0, 0.5, 0.2)
	t.check_near("_decay_exploration returns the new rate", eps.decay_exploration(), 0.5)
	for i in 3: eps.decay_exploration()
	t.check_near("it decays once per call and never goes below its floor", eps.exploration_rate, 0.2)
	var absurd = DTDAQLearning.new(0.1, 0.9, 5.0, 2.0, 1.5)
	absurd.decay_exploration()
	t.check_equal("an exploration rate, a floor and a decay above 1 all stay inside [0, 1]", [absurd.start_exploration_rate, absurd.min_exploration_rate, absurd.exploration_rate], [1.0, 1.0, 1.0])

	t.section("Q-Learning, files, reset and guards (the errors below are expected)")
	var path = "user://dtda_ml_test_qlearning.json"
	for i in TYPED.size(): one.learn("hall", TYPED[i], i + 1, "end", [], true)
	var back = DTDAQLearning.new()
	t.check_equal("_save and _load report a success", [one.save(path), back.load(path)], [true, true])
	t.check_equal("a reloaded agent answers the same policy", [back.predict("a"), back.predict("hall")], [one.predict("a"), one.predict("hall")])
	t.check_near("with the q value it had", back.get_q("a", "go"), one.get_q("a", "go"), 1e-12)
	t.check_equal("and the hyperparameters it had", [back.learning_rate, back.discount_factor, back.exploration_rate], [one.learning_rate, one.discount_factor, one.exploration_rate])
	# a saved table comes back with its keys sorted, so the actions are looked up by their key rather than by the order they were learned in
	var rebuilt = {}
	for action in back._known_actions("hall"): rebuilt[str(action)] = typeof(action)
	for action in TYPED: t.check_equal("the action %s comes back with its type and its value" % action, [rebuilt.size(), rebuilt.get(str(action), TYPE_NIL)], [TYPED.size(), typeof(action)])
	var snapshot = one.to_dict()
	snapshot["q_table"]["a"]["go"] = 999.0
	t.check_near("_to_dict answers a copy of the table, not the table itself", one.get_q("a", "go"), 1.55, 1e-12)
	# a state never met has nothing to answer, actions offered or not, rather than a tie between zeros dressed up as a learned policy
	t.check_equal("_predict on a state the agent never met, even with the actions given", one.predict("nowhere", ACTIONS), null)
	var twin = DTDAQLearning.new(0.1, 0.9, 1.0)
	twin.set_seed(20240817)
	agent.reset()
	t.check_equal("_reset forgets the table and puts the exploration rate back", [agent.q_table, agent.exploration_rate], [null, 1.0])
	for i in 12: t.check_equal("and replays draw %d of a twin seeded the same way" % i, agent.choose_action("s", ACTIONS), twin.choose_action("s", ACTIONS))
	for content in BROKEN:
		FileAccess.open(path, FileAccess.WRITE).store_string(content)
		t.check_equal("_load refuses %s" % content, one.load(path), false)
	t.check_near("and not one of those refusals touched the agent it was handed", one.get_q("a", "go"), 1.55, 1e-12)
