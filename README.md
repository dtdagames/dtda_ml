# dtda_ml
[![tests](https://github.com/dtdagames/dtda_ml/actions/workflows/tests.yml/badge.svg)](https://github.com/dtdagames/dtda_ml/actions/workflows/tests.yml)

DTDA ML allows you to run machine learning models like KNN, Linear Regression, Logistic Regression, SVM


8 models are currently available:
- KNN
- Linear Regression
- Logistic Regression
- SVM
- Decision Tree
- Random Forest
- K-Means
- Q-Learning

The six supervised ones can be scored with the usual metrics. K-Means has no labels to be scored against and reports its inertia instead. Every model can be saved to a JSON file to be reloaded later.


=== Running the tests ===

The repository is also a small Godot project, so you can open it directly and press F6 on addons/dtda_ml/examples/examples_scene.tscn to see every model run.

The test suite lives in tests/ and needs no framework. Run it headless from the project root:
- godot --headless --script res://tests/run_tests.gd

It prints one line per failure and ends with a count, exiting with 0 when everything passes. Some tests exercise the guards of the addon on purpose, so the output contains expected "MLTools: ..." errors: those are not failures.
The same command runs on every push and pull request, against several Godot versions.

Writing a test, two rules that keep the count honest:
- every suite declares "const PLAN = N", the number of assertions it runs. The runner compares it to what it recorded and fails on a mismatch. Add a test, bump the number: the runner tells you which one to write. This is what catches a suite dying halfway through, which would otherwise just show up as a smaller count that nobody reads
- prefer check_equal(name, call(), false) over check(name, not call()). A call that raises a GDScript error answers null, and "not null" is true, so the weak form turns a crash into a pass

check_near() and check_near_array() answer a FAIL when they are handed something that is not a number, rather than letting abs() raise: a model regressed to null is a failure, not a missing line.

=== MLTools features ===

Use MLTools.new() to create a new MLTools. _dropVariable() and _getVariable() allows you to drop a column, or keep column from an array.
This is usefull to create X_train and Y_train for all models

Example:
- data = [
    [1, 1, 1, 0, 1],
    [1, 1, 1, 1, 1],
    [1, 0, 0, 0, 0]
  ]
- var ml = MLTools.new()
- var X_train = ml._dropVariable(data, data[0].size()-1) #return an array of array without the last column
- var y_train = ml._getVariable(data, data[0].size()-1) #return an array of array only with the last column

_get_perf() scores a model: it compares the predictions to the expected labels and returns the percentage of correct answers, rounded to 0.01.
It takes the type of the model as third argument, so it can align both arrays before comparing them:
- 0 : KNN, predictions and labels are compared as they are
- 1 : Linear Regression, predictions above 0.5 are read as 1 and the others as 0, so this only makes sense on a binary target
- 2 : Logistic Regression, predictions and labels are compared as they are
- 3 : SVM, the 0 labels are converted to -1 to match what the model predicts

Example:
- var y_pred = knn._predict(X_test)
- var y_test = [3, 6, 5]
- print("KNN score: ", ml._get_perf(y_pred, y_test, 0), "%")

=== Metrics ===

_get_perf() only knows how to count correct answers. MLTools also carries the usual metrics, which all check the size of both arrays and report an error instead of returning a wrong score.

For classification, the positive label is 1 by default and can be passed as third argument:
- _accuracy(y_pred, y_test) : percentage of correct answers
- _confusion_matrix(y_pred, y_test, positive) : a dictionary with the tp, fp, tn and fn counts
- _precision(y_pred, y_test, positive) : share of the predicted positives that are right, from 0 to 1
- _recall(y_pred, y_test, positive) : share of the real positives that were found, from 0 to 1
- _f1_score(y_pred, y_test, positive) : harmonic mean of precision and recall

For regression, use these rather than _get_perf(), which binarizes at 0.5 and only makes sense on a binary target:
- _mse(y_pred, y_test) and _rmse(y_pred, y_test) : squared error, the RMSE being in the unit of the target
- _mae(y_pred, y_test) : mean absolute error, less sensitive to outliers
- _r2_score(y_pred, y_test) : share of the variance explained, 1.0 is a perfect fit and a model worse than always answering the mean scores below 0

Example:
- var y_pred = logreg._predict(X_test)
- print("F1: ", ml._f1_score(y_pred, y_test))
- print("Confusion: ", ml._confusion_matrix(y_pred, y_test))
- print("R2: ", ml._r2_score(linreg._predict(X_test), y_test))

=== Scaler ===

The models scale their features on their own, so you don't need DTDAScaler to use them. It is there for your own data, and it is what the models use internally.

Use DTDAScaler.new() for a standardization (each column centered on its mean and divided by its standard deviation), or DTDAScaler.new(DTDAScaler.MINMAX) to bring each column into [0, 1]. A constant column is left alone instead of dividing by zero.
_fit() learns the scaling, _transform() applies it, _fit_transform() does both, and _inverse_transform() brings values back to the original unit.

Fit the scaler on your training set only, then apply that same scaler to the test set: refitting on the test set would scale it differently and quietly ruin your predictions.

Example:
- var scaler = DTDAScaler.new()
- var X_train_scaled = scaler._fit_transform(X_train)
- var X_test_scaled = scaler._transform(X_test) #the scaling learned on X_train
- print("Back to the original unit: ", scaler._inverse_transform(X_train_scaled))

=== Saving and loading a model ===

Every model can be written to a JSON file and read back, so you can train once and ship the weights with your game instead of retraining at every launch.
_save(path) returns true on success, _load(path) fills a model you just created. Both report a clear error and return false on failure, and _load() refuses a file holding a different kind of model.

A file lives in user://, where a player can edit it, so _load() checks what it reads before believing it. What is checked is what a prediction computes with: the weights and the intercept of a regression or an SVM, the neighbour count and the training rows of a KNN, the offsets and scales of a scaler, the list of trees of a forest, the centres and the inertia of a K-Means, the q values of an agent. A text where a number belongs, a list that is empty or shorter than the one it goes with, a scale of zero that every prediction would divide by, a neighbour count below one: each of those answers false with an error.

What is not checked, and it is worth knowing: the inside of a decision tree. _load() makes sure the root is a node, then takes the branches as they come, so a tree whose nodes have been rewritten by hand can load and answer nonsense without complaining. Neither is the mode of a forest, which every prediction reads to decide between a vote and a mean: a text there is read as a vote, so a regression forest can come back answering like a classifier, and it does so in silence. Neither are the settings only _fit() reads, such as the learning rate or the number of rounds: a wrong one there costs nothing until you train again.

A file that is refused changes nothing, in any of the eight models: one that was working goes on working, with the weights and the settings it already had.

Use a user:// path: res:// is read only once the game is exported.

Example:
- var linreg = DTDALinReg.new(0.01, 1000)
- linreg._fit(X_train, y_train)
- linreg._save("user://linreg.json")
- var loaded = DTDALinReg.new(0.01, 1000) #a brand new model, never fitted
- if loaded._load("user://linreg.json"):
-  print("Prediction: ", loaded._predict(X_test)) #same results as the saved model

=== KNN Model ===

Use DTDAKNN.new() to create a new model. _fit() and _predict() allows you to train and use the model. This model is better for classification.
The prediction is the most frequent label among the k nearest neighbors. When two labels are tied, the one carried by the closest neighbor wins.

Example:
- var knn = DTDAKNN.new(3)
- knn._fit(X_train, y_train)
- var X_test = [
    [1, 1, 0, 1]
  ]
- print("KNN prediction: ", knn._predict(X_test))

=== Linear Regression Model ===

Use DTDALinReg.new() to create a new model. _fit() and _predict() allows you to train and use the model. This model is better for Regression.
Features and target are standardized internally by _fit(), so the gradient descent stays stable whatever the scale of your data. You don't have to normalize anything beforehand: _predict() gives its results back in the unit of the training target.

Example:
- var linreg = DTDALinReg.new(0.01, 1000)
- linreg._fit(X_train, y_train)
- var X_test = [
    [1, 1, 0, 1]
  ]
- print("Linear Regression prediction: ", linreg._predict(X_test))

=== Logistic Regression Model ===

Use DTDALogReg.new() to create a new model. _fit() and _predict() allows you to train and use the model. This model is only for classification (1 or 0).
Like Linear Regression, the features are standardized internally by _fit(), so you don't have to scale them yourself.

Example:
- var logreg = DTDALogReg.new(0.01, 1000)
- logreg._fit(X_train, y_train)
- var X_test = [
    [1, 1, 0, 1]
  ]
- print("Logistic Regression prediction: ", logreg._predict(X_test))

=== SVM Model ===

Use DTDASVM.new() to create a new model. _fit() and _predict() allows you to train and use the model. This model is only for classification (1 or -1).
The features are standardized internally by _fit() as well. A point sitting exactly on the decision boundary is predicted as 1.

Example:
- var svm = DTDASVM.new(0.01, 0.01, 1000)
- svm._fit(X_train, y_train)
- var X_test = [
    [1, 1, 0, 1]
  ]
- print("SVM prediction: ", svm._predict(X_test))

=== Decision Tree Model ===

Use DTDATree.new() to create a new model. _fit() and _predict() allows you to train and use the model. It is the only model here handling a non linear frontier: it separates a XOR, which the linear models cannot.

DTDATree.new(max_depth, min_samples_split, mode) takes:
- max_depth : how deep the tree may grow, 5 by default. The main guard against overfitting
- min_samples_split : a node holding fewer rows than this becomes a leaf, 2 by default
- mode : DTDATree.CLASSIFIER (the default) splits on the Gini impurity and a leaf answers the majority label, DTDATree.REGRESSOR splits on the variance and a leaf answers the mean
- max_features : how many features a single split may look at, drawn again at every node, 0 by default. 0 means all of them, in order, and draws nothing at all, which is a tree as it has always been. It exists for DTDAForest, which needs its trees to disagree; _set_seed() fixes the draw when you use it yourself

A tree compares each feature to a threshold, so the scale of your data does not matter: unlike the other models it does no scaling at all, and none is needed.
Being made of thresholds, it also answers a constant outside the range it was trained on, where a linear regression keeps extrapolating.

Example:
- var tree = DTDATree.new(3, 2, DTDATree.CLASSIFIER)
- tree._fit(X_train, y_train)
- var X_test = [
    [1, 1, 0, 1]
  ]
- print("Tree prediction: ", tree._predict(X_test))
- var regressor = DTDATree.new(3, 2, DTDATree.REGRESSOR) #same model, on a continuous target
- regressor._fit(X_train, y_train)

=== Random Forest Model ===

Use DTDAForest.new() to create a new forest. It grows a crowd of DTDATree and has them answer together: the majority label when classifying, the mean when regressing. A lone deep tree learns its training set by heart, noise included; a forest cannot, because no two of its trees saw the same thing.

Two draws make the trees disagree, and disagreeing is the whole point, since averaging identical trees gains nothing:
- bagging : each tree is fitted on as many rows as the training set holds, drawn with replacement, so it sees about two thirds of it and a different two thirds
- a feature draw at every split, carried by DTDATree.max_features

DTDAForest.new(num_trees, max_depth, min_samples_split, mode, max_features) takes:
- num_trees : how many trees to grow, 10 by default. More is steadier and slower
- max_depth : how deep each tree may grow, 5 by default. A forest tolerates deeper trees than a lone one, that being what it is for
- min_samples_split : a node holding fewer rows than this becomes a leaf, 2 by default
- mode : DTDAForest.CLASSIFIER (the default) votes, DTDAForest.REGRESSOR averages. They are the modes of DTDATree, taken from it, so the two cannot drift apart
- max_features : how many features a single split may look at, 0 by default. 0 asks for the usual rule, the square root of the count when classifying and a third of it when regressing. Passing the full count turns the forest into plain bagging, which is a fair thing to want and a poor default

There is a draw at every step, so _set_seed(value) is what makes a run repeatable, and _reset() forgets the trees and puts the generator back on that seed. Without a seed, two forests fitted on the same data do not answer quite the same thing.

Like a lone tree, a forest compares features to thresholds and needs no scaling.

Example:
- var forest = DTDAForest.new(25, 8, 2, DTDAForest.CLASSIFIER)
- forest._set_seed(1)
- forest._fit(X_train, y_train)
- print("Forest prediction: ", forest._predict(X_test))
- print("Accuracy: ", ml._accuracy(forest._predict(X_test), y_test), "%")
- forest._save("user://forest.json")

A word on what to expect: on a small test set, a single forest can land level with a single tree by luck, one row being worth a couple of points. The gain is real but it is an average. The examples scene and the test suite both measure it over five seeds rather than one, and so should you.

=== K-Means Model ===

Use DTDAKMeans.new() to create a new model. It is the only one here that is given no labels: _fit(X) is handed rows and nothing else, and works out which of k groups each row belongs to. Reach for it to sort things you have not named yourself, spawn points into territories, players into playstyles, tiles into biomes.

DTDAKMeans.new(k, max_iterations, num_runs) takes:
- k : how many groups to look for, 3 by default. This is yours to choose, the model will not question it
- max_iterations : how many passes one run may take before giving up, 100 by default. A run normally stops on its own, when no row changes group
- num_runs : how many times to start over from a fresh set of centres, 5 by default, keeping the run with the lowest inertia

What you get:
- _predict(X) : the group each row belongs to, as an index from 0 to k-1
- _fit_predict(X) : both at once, on the rows you are fitting
- inertia : the sum of the squared distances from every training row to its centre, left behind by _fit()
- _inertia_of(X) : the same measure for any other rows
- _get_centroids() : the centres, in the unit of the training data, which is what you want to draw
- _set_seed(value) and _reset() : as everywhere else, a run is only repeatable once it has a seed

Distances are euclidean, so a column counted in tens of thousands would drown a column counted in units. The rows are standardised internally, the way DTDALinReg does it: you do not have to scale anything, and multiplying a column by a thousand does not change the answer.

Where the centres start decides where they end, and a poor start stays poor: it is not noise that averages out over the iterations. Both usual answers are built in. k-means++ draws the first centre among the rows, then each of the others with a weight of its squared distance to the nearest centre already chosen, so the starts spread out instead of huddling. On top of that, num_runs starts are tried and the tightest is kept. On four blobs strung along a line, eight runs beat a single one on fifty five seeds out of sixty, and never did worse.

Inertia falls as k rises whatever the grouping is worth, right down to zero when k reaches the number of rows. It cannot be read as a score on its own: it is read across several k on the same data, for where it stops falling sharply.

Example:
- var kmeans = DTDAKMeans.new(3)
- kmeans._set_seed(1)
- print("Groups: ", kmeans._fit_predict(positions))
- print("Centres: ", kmeans._get_centroids())
- for k in range(1, 6): #where does it stop falling
-  var trial = DTDAKMeans.new(k)
-  trial._set_seed(1)
-  trial._fit(positions)
-  print(k, " groups, inertia ", trial.inertia)
- kmeans._save("user://camps.json")

=== Q-Learning Model ===

Use DTDAQLearning.new() to create a new agent. This one is not trained on a dataset: it learns while playing, from the transitions you feed it. It is the model to reach for when you want an NPC that gets better at something instead of a classifier over a CSV.

DTDAQLearning.new(learning_rate, discount_factor, exploration_rate, exploration_decay, min_exploration_rate) takes:
- learning_rate : how much a single transition moves a value, 0.1 by default. Higher learns faster but is noisier
- discount_factor : how much a future reward is worth compared to an immediate one, 0.9 by default. Close to 0 the agent is greedy, close to 1 it plans ahead
- exploration_rate : epsilon, the share of moves taken at random rather than greedily, 1.0 by default so the agent starts by trying everything
- exploration_decay : what epsilon is multiplied by at the end of each episode, 0.99 by default
- min_exploration_rate : the floor epsilon never goes below, 0.01 by default, so the agent keeps a little curiosity

Epsilon is a probability: the rate and its floor are brought into [0, 1] when the agent is built, and _decay_exploration() keeps epsilon in [min_exploration_rate, 1] whatever decay you give it. Nothing else is validated: a discount_factor of 1 or more diverges on a looping world, that one is on you.

The loop:
- _choose_action(state, valid_actions) : epsilon-greedy, picks among the actions that are legal right now. Safe to call before anything was learned, the agent then simply explores. On a state it never met every action is worth 0, so the first of the list comes out
- _learn(state, action, reward, next_state, next_actions, done) : one transition, the Bellman update Q(s, a) += lr * (reward + gamma * max Q(s', a') - Q(s, a)). Pass done = true on the last transition of an episode, a terminal state has no future to add. next_actions restricts what the agent may do next, leave it out (or pass null) to look at everything already learned about next_state
- _decay_exploration() : call it at the end of an episode, epsilon goes down one notch and never below its floor
- _predict(state, valid_actions) : the learned policy with no exploration at all, this is what you ship. valid_actions is optional, leave it out (or pass null) to pick among everything learned in that state. On a state the agent never met, or one whose row holds no action, it reports an error and answers null, with or without a list: it will not dress up a tie between zeros as a policy. Use _choose_action() when you need a move no matter what
- _get_q(state, action) : the value of a pair, 0.0 when it was never met
- _set_seed(value) : fix the random draws for a reproducible run
- _reset() : forget the table, put epsilon back where it started and replay the seed given to _set_seed()

States and actions can be anything, they are not assumed to be contiguous integers: a Vector2i tile, a string, a dictionary key. Both are stored by their str(), which is also what a JSON object needs, so a saved agent comes back keyed exactly the same way. An action is answered back with its own type for int, float, bool, String and StringName.

What keying by str() costs, in exchange:
- two values with the same str() are the same key: the integer 1 and the string "1" share a row, and inside one state the integer 2 and the string "2" share a cell. Use one spelling per state and per action
- which pairs collide follows the number formatting of the engine, and that moves between versions: str(2.0) prints "2" up to Godot 4.3 and "2.0" from 4.4 on, so a whole float and an integer are one key on 4.3 and two different ones on 4.4. Do not rely on either behaviour
- across two states, colliding actions keep their own values but not their type: the type of an action is remembered once for the whole agent, so the last one learned decides what _predict() hands back everywhere
- a float key goes through str(), which may not carry every digit of a double: depending on the engine version, 1.0/3.0 comes back exactly or a hair off. Quantize a continuous state, a position for instance, rather than rely on either
- for the same reason, a file holding float keys is tied to the engine it was written on: an agent saved on 4.3 with the state 2.0 wrote the key "2", which 4.4 spells "2.0" and would no longer find. Integer and string keys are unaffected

A saved agent carries the version of the format it was written in, and _load() refuses anything else rather than reading it wrong.

Example:
- var agent = DTDAQLearning.new(0.2, 0.9, 1.0, 0.999, 0.1)
- for episode in 1000:
-  var state = start_state
-  while not done:
-   var action = agent._choose_action(state, ["left", "right"])
-   #play the action, get the reward and the next state from your game
-   agent._learn(state, action, reward, next_state, ["left", "right"], done)
-   state = next_state
-   agent._decay_exploration()
- agent._save("user://agent.json")
- print("Best move: ", agent._predict(state)) #no exploration left, the learned policy
