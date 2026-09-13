extends DTDATools

class_name DTDAQLearning

# Tabular Q-learning: no training set, the agent improves from the transitions it is fed.
# States and actions are free values, keyed by str() so a table survives JSON unchanged: two
# values sharing a str() share a cell (integer 1 and string "1"), and str() of a float moves
# between engines (2.0 prints "2" up to 4.3, "2.0" from 4.4), tying a file with float keys to
# the engine that wrote it, so quantize a continuous state. An action comes back with its own type when it can, otherwise as its string key, and that one warns when saved.

# version 1 wrote the type of an action as the raw engine enum value, version 2 the stable labels below, and content alone cannot tell them apart: an unversioned file is refused
const FORMAT_VERSION = 2

const ACTION_TYPES = {
	TYPE_INT: "int",
	TYPE_FLOAT: "float",
	TYPE_BOOL: "bool",
	TYPE_STRING: "string",
	TYPE_STRING_NAME: "string_name",
}

var learning_rate: float
var discount_factor: float
var exploration_rate: float
var exploration_decay: float
var min_exploration_rate: float
var start_exploration_rate: float
var q_table
# {action key: the action itself}, global to the agent rather than kept per state
var actions_seen
var rng: RandomNumberGenerator
var start_seed

func _init(q_learning_rate: float = 0.1, q_discount_factor: float = 0.9, q_exploration_rate: float = 1.0, q_exploration_decay: float = 0.99, q_min_exploration_rate: float = 0.01) -> void:
	learning_rate = q_learning_rate
	discount_factor = q_discount_factor
# epsilon and its floor are probabilities, clamped here rather than left to contradict it
	exploration_rate = clamp(q_exploration_rate, 0.0, 1.0)
	start_exploration_rate = exploration_rate
	exploration_decay = q_exploration_decay
	min_exploration_rate = clamp(q_min_exploration_rate, 0.0, 1.0)
	actions_seen = {}
	rng = RandomNumberGenerator.new()
	start_seed = null

func _key(value) -> String:
	return str(value)

# fix the random draws for a reproducible run; reset() replays this same seed
func set_seed(value: int) -> void:
	start_seed = value
	rng.seed = value

func _as_list(valid_actions) -> Array:
	if valid_actions == null:
		return []
	return valid_actions

func get_q(state, action) -> float:
	if q_table == null:
		return 0.0
	var row = q_table.get(_key(state))
	if row == null:
		return 0.0
	return row.get(_key(action), 0.0)

func _known_actions(state) -> Array:
	var known: Array = []
	if q_table == null:
		return known
	var row = q_table.get(_key(state))
	if row == null:
		return known
	for action_key in row:
		known.push_back(actions_seen.get(action_key, action_key))
	return known

func _max_q(state, valid_actions = []) -> float:
	var candidates = _as_list(valid_actions)
	if candidates.is_empty():
		candidates = _known_actions(state)
	if candidates.is_empty():
		return 0.0
	var best: float = -INF
	for action in candidates:
		var value = get_q(state, action)
		if value > best:
			best = value
	return best

# strict comparison, so a tie keeps the first action and an agent that knows nothing yet stays reproducible
func _best_action(state, valid_actions):
	var best = null
	var best_value: float = -INF
	for action in valid_actions:
		var value = get_q(state, action)
		if value > best_value:
			best = action
			best_value = value
	return best

# epsilon-greedy; on a state never met every q is 0, a tie, so the first action of the list comes out
func choose_action(state, valid_actions):
	if valid_actions == null or valid_actions.size() == 0:
		push_error("DTDAQLearning: choose_action() called without any valid action")
		return null
# randf() lives in [0, 1), so an exploration_rate of 0.0 never explores and 1.0 always does
	if rng.randf() < exploration_rate:
		return valid_actions[rng.randi() % valid_actions.size()]
	return _best_action(state, valid_actions)

# Bellman: Q(s, a) += lr * (reward + gamma * max Q(s', a') - Q(s, a)); a terminal transition has no future, its target is the reward alone
func learn(state, action, reward, next_state, next_actions = [], done = false):
# a nan reward answers false to every comparison, leaving predict() unable to name a best action here: refuse, answer null, keep the cell
	if not _check_number(reward, "DTDAQLearning", "reward"):
		return null
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

func predict(state, valid_actions = []):
	if not _check_fitted("DTDAQLearning", q_table):
		return null
# a state never met would hand back the first action offered dressed as a learned policy; choose_action() is the one that answers regardless
	if not q_table.has(_key(state)):
		push_error("DTDAQLearning: predict() knows nothing about the state '%s'" % _key(state))
		return null
	var candidates = _as_list(valid_actions)
	if candidates.is_empty():
		candidates = _known_actions(state)
	# a row can be there and hold nothing, a file where a state was emptied by hand
	if candidates.is_empty():
		push_error("DTDAQLearning: predict() knows no action for the state '%s'" % _key(state))
		return null
	return _best_action(state, candidates)

# epsilon stays in [min_exploration_rate, 1] whatever the decay
func decay_exploration() -> float:
	exploration_rate = max(min_exploration_rate, exploration_rate * exploration_decay)
	exploration_rate = clamp(exploration_rate, 0.0, 1.0)
	return exploration_rate

func reset() -> void:
	q_table = null
	actions_seen = {}
	exploration_rate = start_exploration_rate
	if start_seed != null:
		rng.seed = start_seed

# an action goes out with the label of its type, so the integer 2 does not come back as "2" or 2.0
func _actions_to_dict() -> Dictionary:
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
		_:
			return action_key

func to_dict() -> Dictionary:
	if not _check_fitted("DTDAQLearning", q_table, "save()"):
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

func from_dict(data) -> bool:
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
# a model file lives in user://, where a player can edit it: rebuild row by row and only replace the table once it is sound
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


# the older underscored spellings, kept working for what already calls them; they only forward

func _set_seed(value):
	set_seed(value)

func _get_q(state, action):
	return get_q(state, action)

func _choose_action(state, valid_actions):
	return choose_action(state, valid_actions)

func _learn(state, action, reward, next_state, next_actions = [], done = false):
	return learn(state, action, reward, next_state, next_actions, done)

func _predict(state, valid_actions = []):
	return predict(state, valid_actions)

func _decay_exploration():
	return decay_exploration()

func _reset():
	reset()

