extends MLTools

class_name DTDAQLearning

# === Q-Learning === #
# A tabular agent that learns while playing: there is no training set, it improves from
# the transitions you feed it. The table holds the expected discounted return of every
# (state, action) pair met so far, and the policy is "take the best action of this state".
#
# States and actions are free, they are not assumed to be contiguous integers. Both are
# keyed by str(), which is also what a JSON object needs, so a table written to disk comes
# back usable with the very same native values. What that costs:
#  - two values with the same str() are the same key: the integer 1 and the string "1"
#    share a row, and inside one state the integer 2 and the string "2" share a cell.
#    Which pairs collide follows the number formatting of the engine, and that does
#    move: str(2.0) prints "2" up to Godot 4.3 and "2.0" from 4.4 on, so a whole float
#    and an integer are one key on 4.3 and two on 4.4. Use one spelling per action
#  - across two states colliding actions keep their own q values, but not their type:
#    it is remembered once for the whole agent, so the last one learned decides what
#    _predict() hands back everywhere, in every state
#  - a float key goes through str(), which may not carry every digit of a double:
#    depending on the engine, 1.0/3.0 comes back exactly or a hair off. Quantize a
#    continuous state, a position for instance, rather than rely on either
#  - a file holding float keys is tied to the engine it was written on, for the same
#    reason: an agent saved on 4.3 with the state 2.0 wrote the key "2", which 4.4
#    spells "2.0" and would no longer find. Integer and string keys are unaffected
# An action is answered back with its own type for int, float, bool, String and
# StringName, anything else comes back as its string key and warns when saved.

# version 1 wrote the type of an action as the raw value of the engine enum, which
# version 2 replaced by the stable labels below. The two cannot be told apart from
# the content alone, so a file that does not announce the current version is refused
const FORMAT_VERSION = 2

# a stable label per type, rather than the raw value of the engine enum, so a file
# written today still reads the same way on a later version of Godot
const ACTION_TYPES = {
	TYPE_INT: "int",
	TYPE_FLOAT: "float",
	TYPE_BOOL: "bool",
	TYPE_STRING: "string",
	TYPE_STRING_NAME: "string_name",
}

var learning_rate
var discount_factor
# epsilon: the share of the moves taken at random rather than greedily
var exploration_rate
var exploration_decay
var min_exploration_rate
# where the exploration started, so _reset() puts the agent back as it was
var start_exploration_rate
# {state key: {action key: q value}}, null until the first _learn()
var q_table
# {action key: the action itself}, so the agent answers with what you passed it
# it is global to the agent, not kept per state
var actions_seen
# its own generator, so a game can replay an identical run with _set_seed()
var rng
# the seed given to _set_seed(), replayed by _reset(), null when none was asked for
var start_seed

func _init(q_learning_rate := 0.1, q_discount_factor := 0.9, q_exploration_rate := 1.0, q_exploration_decay := 0.99, q_min_exploration_rate := 0.01):
	learning_rate = q_learning_rate
	discount_factor = q_discount_factor
	# epsilon and its floor are probabilities, they are brought into [0, 1] here
	# rather than left to contradict the interval the rest of the class promises
	exploration_rate = clamp(q_exploration_rate, 0.0, 1.0)
	start_exploration_rate = exploration_rate
	exploration_decay = q_exploration_decay
	min_exploration_rate = clamp(q_min_exploration_rate, 0.0, 1.0)
	actions_seen = {}
	rng = RandomNumberGenerator.new()
	start_seed = null

# a JSON object only has string keys, so the table is keyed by str() from the start:
# what is written is exactly what is read back
func _key(value):
	return str(value)

# fix the random draws, for a reproducible training run
# _reset() puts the generator back on that same seed
func _set_seed(value):
	start_seed = value
	rng.seed = value

# an omitted or null list of actions means "everything already known here"
func _as_list(valid_actions):
	if valid_actions == null:
		return []
	return valid_actions

# expected return of an action in a state, 0.0 when it was never met
func _get_q(state, action):
	if q_table == null:
		return 0.0
	var row = q_table.get(_key(state))
	if row == null:
		return 0.0
	return row.get(_key(action), 0.0)

# the actions already met in a state, in the order they were first learned
func _known_actions(state):
	var known = []
	if q_table == null:
		return known
	var row = q_table.get(_key(state))
	if row == null:
		return known
	for action_key in row:
		known.push_back(actions_seen.get(action_key, action_key))
	return known

# best q value reachable from a state, 0.0 when nothing is known about it
func _max_q(state, valid_actions = []):
	var candidates = _as_list(valid_actions)
	if candidates.is_empty():
		candidates = _known_actions(state)
	if candidates.is_empty():
		return 0.0
	var best = -INF
	for action in candidates:
		var value = _get_q(state, action)
		if value > best:
			best = value
	return best

# the action with the highest q value among the given ones
# the comparison is strict, so a tie keeps the first action of the list and the
# answer of an agent that knows nothing yet stays reproducible
func _best_action(state, valid_actions):
	var best = null
	var best_value = -INF
	for action in valid_actions:
		var value = _get_q(state, action)
		if value > best_value:
			best = action
			best_value = value
	return best

# epsilon-greedy: exploration_rate of the time a random valid action, the rest of the
# time the best one known so far. This is the one that may answer on a state it never
# met, which is the whole point of the first episode: every q value is 0, a tie, so the
# first action of the list comes out
func _choose_action(state, valid_actions):
	if valid_actions == null or valid_actions.size() == 0:
		push_error("DTDAQLearning: _choose_action() called without any valid action")
		return null
	# randf() lives in [0, 1), so an exploration_rate of 0.0 never explores
	# and one of 1.0 always does
	if rng.randf() < exploration_rate:
		return valid_actions[rng.randi() % valid_actions.size()]
	return _best_action(state, valid_actions)

# one transition, the Bellman update:
#   Q(s, a) += lr * (reward + gamma * max Q(s', a') - Q(s, a))
# a terminal transition has no future, its target is the reward alone
# next_actions restricts what the agent may do next, leave it out to look at
# everything already known about next_state
func _learn(state, action, reward, next_state, next_actions = [], done = false):
	if q_table == null:
		q_table = {}
	var state_key = _key(state)
	var action_key = _key(action)
	if not q_table.has(state_key):
		q_table[state_key] = {}
	actions_seen[action_key] = action
	var current = q_table[state_key].get(action_key, 0.0)
	var target = reward
	if not done:
		target += discount_factor * _max_q(next_state, next_actions)
	q_table[state_key][action_key] = current + learning_rate * (target - current)
	return q_table[state_key][action_key]

# the learned policy, without any exploration: the best action known for this state
# valid_actions restricts the choice, leave it out to pick among everything learned there
func _predict(state, valid_actions = []):
	if not _check_fitted("DTDAQLearning", q_table):
		return null
	# a state never met has nothing to answer, whatever the actions offered: they would
	# all be worth 0 and the first of the list would come out dressed as a learned
	# policy. Use _choose_action() when you need a move no matter what
	if not q_table.has(_key(state)):
		push_error("DTDAQLearning: _predict() knows nothing about the state '%s'" % _key(state))
		return null
	var candidates = _as_list(valid_actions)
	if candidates.is_empty():
		candidates = _known_actions(state)
	# a row can be there and hold nothing, a file where a state was emptied by hand
	if candidates.is_empty():
		push_error("DTDAQLearning: _predict() knows no action for the state '%s'" % _key(state))
		return null
	return _best_action(state, candidates)

# call at the end of an episode: the agent explores a little less from now on
# epsilon is a probability, it stays in [min_exploration_rate, 1] whatever the decay
func _decay_exploration():
	exploration_rate = max(min_exploration_rate, exploration_rate * exploration_decay)
	exploration_rate = clamp(exploration_rate, 0.0, 1.0)
	return exploration_rate

# forget everything learned and put the agent back where it started: the exploration
# rate it was built with, and the seed it was given
func _reset():
	q_table = null
	actions_seen = {}
	exploration_rate = start_exploration_rate
	if start_seed != null:
		rng.seed = start_seed

# an action goes out as its key plus the label of its type, so the integer 2 does not
# come back as the string "2" or as the float 2.0
func _actions_to_dict():
	var types = {}
	for action_key in actions_seen:
		var type = typeof(actions_seen[action_key])
		if not ACTION_TYPES.has(type):
			push_warning("DTDAQLearning: the action '%s' is not a scalar, it will be read back as a string" % action_key)
		types[action_key] = ACTION_TYPES.get(type, "string")
	return types

func _action_from_key(action_key, label):
	match label:
		"int":
			return int(action_key)
		"float":
			return float(action_key)
		"bool":
			return action_key == "true"
		"string_name":
			return StringName(action_key)
		# a string comes back as its own key, and so does anything we cannot rebuild
		_:
			return action_key

func _to_dict():
	if not _check_fitted("DTDAQLearning", q_table, "_save()"):
		return {}
	return {
		"model": "DTDAQLearning",
		"version": FORMAT_VERSION,
		"learning_rate": learning_rate,
		"discount_factor": discount_factor,
		"exploration_rate": exploration_rate,
		"start_exploration_rate": start_exploration_rate,
		"exploration_decay": exploration_decay,
		"min_exploration_rate": min_exploration_rate,
		# a deep copy: the caller gets a snapshot, not the table the agent keeps learning on
		"q_table": q_table.duplicate(true),
		"actions": _actions_to_dict(),
	}

func _from_dict(data):
	if not _check_model_name(data, "DTDAQLearning"):
		return false
	# int() because a version read back from JSON carries as a float
	var version = int(data.get("version", 0))
	if version != FORMAT_VERSION:
		push_error("DTDAQLearning: this file is written in format %d, this model reads format %d" % [version, FORMAT_VERSION])
		return false
	var table = data.get("q_table")
	if table == null:
		push_error("DTDAQLearning: the saved model has no q table")
		return false
	if typeof(table) != TYPE_DICTIONARY:
		push_error("DTDAQLearning: the saved q table is not a table")
		return false
	# a model file lives in user://, where a player can edit it: the table is rebuilt
	# row by row and only replaces the current one once it is known to be sound
	var rebuilt = {}
	for state_key in table:
		var row = table[state_key]
		if typeof(row) != TYPE_DICTIONARY:
			push_error("DTDAQLearning: the saved state '%s' holds no action" % state_key)
			return false
		var values = {}
		for action_key in row:
			var value = row[action_key]
			if not (typeof(value) in [TYPE_INT, TYPE_FLOAT]):
				push_error("DTDAQLearning: the saved q value of '%s' is not a number" % action_key)
				return false
			values[action_key] = float(value)
		rebuilt[state_key] = values

	learning_rate = float(data.get("learning_rate", learning_rate))
	discount_factor = float(data.get("discount_factor", discount_factor))
	# a file can be edited by hand, epsilon stays in [0, 1] whatever it says
	exploration_rate = clamp(float(data.get("exploration_rate", exploration_rate)), 0.0, 1.0)
	start_exploration_rate = clamp(float(data.get("start_exploration_rate", exploration_rate)), 0.0, 1.0)
	exploration_decay = float(data.get("exploration_decay", exploration_decay))
	min_exploration_rate = clamp(float(data.get("min_exploration_rate", min_exploration_rate)), 0.0, 1.0)
	q_table = rebuilt
	actions_seen = {}
	var types = data.get("actions", {})
	if typeof(types) != TYPE_DICTIONARY:
		types = {}
	for action_key in types:
		actions_seen[action_key] = _action_from_key(action_key, types[action_key])
	return true

# === End Q-Learning === #
