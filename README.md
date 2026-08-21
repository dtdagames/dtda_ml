# dtda_ml
[![tests](https://github.com/dtdagames/dtda_ml/actions/workflows/tests.yml/badge.svg)](https://github.com/dtdagames/dtda_ml/actions/workflows/tests.yml)

Machine learning for Godot games, written in GDScript with no dependencies. It is a library rather than an editor plugin: copy `addons/dtda_ml` into your project and the classes are there, with nothing to enable.

There are two things people actually do with it.

**Train elsewhere, ship the weights, predict at runtime.** Fit a model wherever it is convenient, save it to JSON, and load it in the game. This is the sensible route for the models that learn by gradient descent: fitting a regression inside your game is rarely what you want, loading one that is already trained is.

- var aim = DTDALinReg.new(0.01, 1000)
- if aim.load("res://weights/aim.json"):
-     print(aim.predict([[distance, speed]]))

**Learn while the game runs.** DTDAQLearning is the one model that has to live in the engine, because an agent that learns by playing cannot be trained anywhere else. It takes one transition at a time, a few microseconds each, so it costs nothing per frame.

- var move = agent.choose_action(state, ["left", "right"])
- #play the move, then tell the agent how it went
- agent.learn(state, move, reward, next_state, ["left", "right"], done)

Training in the game is possible for the others too, without freezing a frame: see "Training a slice at a time".

8 models are currently available:
- KNN
- Linear Regression
- Logistic Regression
- SVM
- Decision Tree
- Random Forest
- K-Means
- Q-Learning

Every method used to carry a leading underscore, which in Godot marks a method as virtual or private. They have lost it: it is fit(), predict(), save(), load(). The older spellings still work, so nothing already written breaks, but prefer the ones without. The toolbox is DTDATools now rather than MLTools, and MLTools still answers as well. One thing does change: the models extend DTDATools, so "model is MLTools" is false where it used to be true.

The six supervised ones can be scored with the usual metrics. K-Means has no labels to be scored against and reports its inertia instead. Every model can be saved to a JSON file to be reloaded later.


=== Running the tests ===

The repository is also a small Godot project, so you can open it directly and press F6 on addons/dtda_ml/examples/examples_scene.tscn to see every model run.

The test suite lives in tests/ and needs no framework. Run it headless from the project root:
- godot --headless --script res://tests/run_tests.gd

It prints one line per failure and ends with a count, exiting with 0 when everything passes. Some tests exercise the guards of the addon on purpose, so the output contains errors prefixed with the name of the model that raised them, "DTDAKMeans: ..." and the like: those are expected and are not failures.
The same command runs on every push and pull request, against several Godot versions.

Writing a test, two rules that keep the count honest:
- every suite declares "const PLAN = N", the number of assertions it runs. The runner compares it to what it recorded and fails on a mismatch. Add a test, bump the number: the runner tells you which one to write. This is what catches a suite dying halfway through, which would otherwise just show up as a smaller count that nobody reads
- prefer check_equal(name, call(), false) over check(name, not call()). A call that raises a GDScript error answers null, and "not null" is true, so the weak form turns a crash into a pass

check_near() and check_near_array() answer a FAIL when they are handed something that is not a number, rather than letting abs() raise: a model regressed to null is a failure, not a missing line.

=== DTDATools features ===

Use DTDATools.new() to create one. drop_variable() and get_variable() allows you to drop a column, or keep column from an array.
This is usefull to create X_train and Y_train for all models

Example:
- data = [
    [1, 1, 1, 0, 1],
    [1, 1, 1, 1, 1],
    [1, 0, 0, 0, 0]
  ]
- var ml = DTDATools.new()
- var X_train = ml.drop_variable(data, data[0].size()-1) #return an array of array without the last column
- var y_train = ml.get_variable(data, data[0].size()-1) #return an array of array only with the last column

get_perf() scores a model: it compares the predictions to the expected labels and returns the percentage of correct answers, rounded to 0.01.
It takes the type of the model as third argument, so it can align both arrays before comparing them:
- 0 : KNN, predictions and labels are compared as they are
- 1 : Linear Regression, predictions above 0.5 are read as 1 and the others as 0, so this only makes sense on a binary target
- 2 : Logistic Regression, predictions and labels are compared as they are
- 3 : SVM, the 0 labels are converted to -1 to match what the model predicts

Example:
- var y_pred = knn.predict(X_test)
- var y_test = [3, 6, 5]
- print("KNN score: ", ml.get_perf(y_pred, y_test, 0), "%")

=== Metrics ===

get_perf() only knows how to count correct answers. DTDATools also carries the usual metrics, which all check the size of both arrays and report an error instead of returning a wrong score.

For classification, the positive label is 1 by default and can be passed as third argument:
- accuracy(y_pred, y_test) : percentage of correct answers
- confusion_matrix(y_pred, y_test, positive) : a dictionary with the tp, fp, tn and fn counts
- precision(y_pred, y_test, positive) : share of the predicted positives that are right, from 0 to 1
- recall(y_pred, y_test, positive) : share of the real positives that were found, from 0 to 1
- f1_score(y_pred, y_test, positive) : harmonic mean of precision and recall

For regression, use these rather than get_perf(), which binarizes at 0.5 and only makes sense on a binary target:
- mse(y_pred, y_test) and rmse(y_pred, y_test) : squared error, the RMSE being in the unit of the target
- mae(y_pred, y_test) : mean absolute error, less sensitive to outliers
- r2_score(y_pred, y_test) : share of the variance explained, 1.0 is a perfect fit and a model worse than always answering the mean scores below 0

Example:
- var y_pred = logreg.predict(X_test)
- print("F1: ", ml.f1_score(y_pred, y_test))
- print("Confusion: ", ml.confusion_matrix(y_pred, y_test))
- print("R2: ", ml.r2_score(linreg.predict(X_test), y_test))

=== Scaler ===

The models scale their features on their own, so you don't need DTDAScaler to use them. It is there for your own data, and it is what the models use internally.

Use DTDAScaler.new() for a standardization (each column centered on its mean and divided by its standard deviation), or DTDAScaler.new(DTDAScaler.MINMAX) to bring each column into [0, 1]. A constant column is left alone instead of dividing by zero.
fit() learns the scaling, transform() applies it, fit_transform() does both, and inverse_transform() brings values back to the original unit.

Fit the scaler on your training set only, then apply that same scaler to the test set: refitting on the test set would scale it differently and quietly ruin your predictions.

Example:
- var scaler = DTDAScaler.new()
- var X_train_scaled = scaler.fit_transform(X_train)
- var X_test_scaled = scaler.transform(X_test) #the scaling learned on X_train
- print("Back to the original unit: ", scaler.inverse_transform(X_train_scaled))

=== What a fit accepts ===

fit() is handed whatever your own code computed, and one unlucky division upstream is enough to hand it a nan. A nan answers false to every comparison, so it used to travel into the weights of a regression and stay there, every prediction answering nan from then on without a word.

So fit() weighs the rows before it writes anything down, and answers true when it fitted and false when it refused:
- the rows have to be a list of rows, none of them empty, all of the same width, holding only real numbers. A text, a nan or an infinity in any cell is refused
- there have to be as many labels as rows

What the labels hold is left alone where the model only counts them or hands them back: a KNN answers a label as it came, and a tree or a forest classifying only counts it, so a label can be a string naming a class, and passing "cave" and "camp" to any of the three is supported rather than tolerated. The three models that descend on a label, and a tree or a forest in REGRESSOR mode, do weigh it, because there it is a number they compute with.

A fit that is refused changes nothing: a model that was working goes on working, with the weights it already had. That holds for the seven models that have a fit(); DTDAQLearning learns one transition at a time instead, and learn() holds the same promise, answering null and leaving the cell as it was when the reward is not a real number. It is the same promise as the one on load(), for the same reason, and it is the one that matters more, since no file has to be edited for a caller to hand over a nan.

Example:
- if not model.fit(X_train, y_train):
-     print("the training data was not usable, the model is untouched")

=== Training a slice at a time ===

fit() runs to the end before it returns. On two hundred rows of eight columns, a forest of 25 trees takes the better part of a second, and several seconds on data its columns explain poorly: a frozen frame, dozens of them in a row, and a game cannot afford one.

The same training can be taken a slice at a time instead, one call per frame:

- if forest.fit_begin(X, y):
-     while forest.is_fitting():
-         var done: float = forest.fit_step()   #0.0 to 1.0, a progress bar if you want one
-         await get_tree().process_frame

fit() is that loop and nothing else, so a model trained in one go is the same model, to the last bit, as one stepped by hand. Existing code that calls fit() is unaffected.

Five models take slices, each in the unit that suits it. On two hundred rows of eight columns whose labels the columns explain in part, the longest single slice came out roughly:

- DTDALinReg, one round of the descent : a couple of ms
- DTDALogReg, one round of the descent : a couple of ms
- DTDASVM, one pass over the rows : a few ms
- DTDAKMeans, one run from fresh starts : a few ms
- DTDAForest, one tree : some tens of ms

**Those are orders of magnitude, and the ordering between them is what to rely on.** Two things move them, and only one is under your control.

The first is the machine, and it moves them further than you would expect. The very same slice, on the very same data, repeated inside a single run, came out between three and thirteen milliseconds: a factor of four, on work that does not vary at all. The figures above are at or below the favourable end of that spread, taken on a quiet machine; a busy one runs several times slower again, and a slower machine several times more. Every number on this page is one machine on one afternoon.

The second is the nature of the data, and it moves the last two only. A round of gradient descent is a matrix pass of a size the data cannot change, so the three descents cost what they cost whatever the labels say. A tree and a Lloyd run stop when they run out of work, so what they cost depends on how well the columns explain the labels. On that same two hundred by eight, with nothing altered but the labels, the same forest with the same seed ran **an order of magnitude apart on identical shape**: quickest when one of the eight columns cut the labels cleanly, slowest when nothing in the columns explained them at all. How far apart depends on how separable the data is, and it can be several times more, or several times less. Expect the noisy end to cost you that much over the table, and measure your own.

Whether any of them fits inside a frame is a question about your machine, not about this table: on the one these were taken on the three descents came in under a frame and the other two did not, and on a machine two or three times slower none of the five did. **What holds is the ordering.** If a slice has to fit in a frame, the descents are the ones that might and the forest the last one that will. **And none of this scales indefinitely, so it is worth knowing where it stops.** On a thousand rows of twelve columns the descents run into the tens of milliseconds a slice, at or past a frame, and the two that depend on the data go far beyond. The forest cannot be cut finer than one tree without DTDATree itself becoming resumable, which it is not.

DTDAKNN and DTDATree train in one go and have no fit_begin(). They are the two that need it least: fitting a KNN on a thousand rows is single-digit milliseconds, since it only keeps them.

A training that never finishes changes nothing. Nothing is written into the model until the last slice, so a training that is abandoned, or stopped with fit_cancel(), or refused by fit_begin() because the rows are not usable, leaves the model it was going to replace exactly as it was. Either the old one whole or the new one whole, never a half of each. It is the promise that governs a refused file and a refused fit, applied to time.

is_fitting(), fit_step() and fit_cancel() are safe to call on any model: one that trains in one go reports an error and answers that there is nothing left to do, rather than looping for ever.

=== Saving and loading a model ===

Every model can be written to a JSON file and read back, so you can train once and ship the weights with your game instead of retraining at every launch.
save(path) returns true on success, load(path) fills a model you just created. Both report a clear error and return false on failure, and load() refuses a file holding a different kind of model.

A file lives in user://, where a player can edit it, so load() checks what it reads before believing it. What is checked is what a prediction computes with: the weights and the intercept of a regression or an SVM, the neighbour count and the training rows of a KNN, the offsets and scales of a scaler, the list of trees of a forest, the centres and the inertia of a K-Means, the q values of an agent. A text where a number belongs, an infinity where a number to compute with belongs, a list that is empty or shorter than the one it goes with, a scale of zero that every prediction would divide by, a neighbour count below one: each of those answers false with an error. The infinity is not hypothetical here: a weight written 1e400 in the file is more than a float can hold and comes back from JSON as an infinity. A literal nan or inf, which is what save() writes if a model ever holds one, makes the whole file unreadable and is turned away a step earlier.

What is not checked, and it is worth knowing: the inside of a decision tree. load() makes sure the root is a node, then takes the branches as they come, so a tree whose nodes have been rewritten by hand can load and answer nonsense without complaining. Neither is the mode of a forest, which every prediction reads to decide between a vote and a mean: a text there is read as a vote, so a regression forest can come back answering like a classifier, and it does so in silence. Neither are the settings only fit() reads, such as the learning rate or the number of rounds: a wrong one there costs nothing until you train again.

A file that is refused changes nothing, in any of the eight models: one that was working goes on working, with the weights and the settings it already had.

Use a user:// path: res:// is read only once the game is exported.

Example:
- var linreg = DTDALinReg.new(0.01, 1000)
- linreg.fit(X_train, y_train)
- linreg.save("user://linreg.json")
- var loaded = DTDALinReg.new(0.01, 1000) #a brand new model, never fitted
- if loaded.load("user://linreg.json"):
-  print("Prediction: ", loaded.predict(X_test)) #same results as the saved model

=== KNN Model ===

Use DTDAKNN.new() to create a new model. fit() and predict() allows you to train and use the model. This model is better for classification.
The prediction is the most frequent label among the k nearest neighbors. When two labels are tied, the one carried by the closest neighbor wins.

Example:
- var knn = DTDAKNN.new(3)
- knn.fit(X_train, y_train)
- var X_test = [
    [1, 1, 0, 1]
  ]
- print("KNN prediction: ", knn.predict(X_test))

=== Linear Regression Model ===

Use DTDALinReg.new() to create a new model. fit() and predict() allows you to train and use the model. This model is better for Regression.
Features and target are standardized internally by fit(), so the gradient descent stays stable whatever the scale of your data. You don't have to normalize anything beforehand: predict() gives its results back in the unit of the training target.

Example:
- var linreg = DTDALinReg.new(0.01, 1000)
- linreg.fit(X_train, y_train)
- var X_test = [
    [1, 1, 0, 1]
  ]
- print("Linear Regression prediction: ", linreg.predict(X_test))

=== Logistic Regression Model ===

Use DTDALogReg.new() to create a new model. fit() and predict() allows you to train and use the model. This model is only for classification (1 or 0).
Like Linear Regression, the features are standardized internally by fit(), so you don't have to scale them yourself.

Example:
- var logreg = DTDALogReg.new(0.01, 1000)
- logreg.fit(X_train, y_train)
- var X_test = [
    [1, 1, 0, 1]
  ]
- print("Logistic Regression prediction: ", logreg.predict(X_test))

=== SVM Model ===

Use DTDASVM.new() to create a new model. fit() and predict() allows you to train and use the model. This model is only for classification (1 or -1).
The features are standardized internally by fit() as well. A point sitting exactly on the decision boundary is predicted as 1.

Example:
- var svm = DTDASVM.new(0.01, 0.01, 1000)
- svm.fit(X_train, y_train)
- var X_test = [
    [1, 1, 0, 1]
  ]
- print("SVM prediction: ", svm.predict(X_test))

=== Decision Tree Model ===

Use DTDATree.new() to create a new model. fit() and predict() allows you to train and use the model. It is the only model here handling a non linear frontier: it separates a XOR, which the linear models cannot.

DTDATree.new(max_depth, min_samples_split, mode) takes:
- max_depth : how deep the tree may grow, 5 by default. The main guard against overfitting
- min_samples_split : a node holding fewer rows than this becomes a leaf, 2 by default
- mode : DTDATree.CLASSIFIER (the default) splits on the Gini impurity and a leaf answers the majority label, DTDATree.REGRESSOR splits on the variance and a leaf answers the mean
- max_features : how many features a single split may look at, drawn again at every node, 0 by default. 0 means all of them, in order, and draws nothing at all, which is a tree as it has always been. It exists for DTDAForest, which needs its trees to disagree; set_seed() fixes the draw when you use it yourself

A tree compares each feature to a threshold, so the scale of your data does not matter: unlike the other models it does no scaling at all, and none is needed.
Being made of thresholds, it also answers a constant outside the range it was trained on, where a linear regression keeps extrapolating.

Example:
- var tree = DTDATree.new(3, 2, DTDATree.CLASSIFIER)
- tree.fit(X_train, y_train)
- var X_test = [
    [1, 1, 0, 1]
  ]
- print("Tree prediction: ", tree.predict(X_test))
- var regressor = DTDATree.new(3, 2, DTDATree.REGRESSOR) #same model, on a continuous target
- regressor.fit(X_train, y_train)

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

There is a draw at every step, so set_seed(value) is what makes a run repeatable, and reset() forgets the trees and puts the generator back on that seed. Without a seed, two forests fitted on the same data do not answer quite the same thing.

Like a lone tree, a forest compares features to thresholds and needs no scaling.

Example:
- var forest = DTDAForest.new(25, 8, 2, DTDAForest.CLASSIFIER)
- forest.set_seed(1)
- forest.fit(X_train, y_train)
- print("Forest prediction: ", forest.predict(X_test))
- print("Accuracy: ", ml.accuracy(forest.predict(X_test), y_test), "%")
- forest.save("user://forest.json")

A word on what to expect: on a small test set, a single forest can land level with a single tree by luck, one row being worth a couple of points. The gain is real but it is an average. The examples scene and the test suite both measure it over five seeds rather than one, and so should you.

=== K-Means Model ===

Use DTDAKMeans.new() to create a new model. It is the only one here that is given no labels: fit(X) is handed rows and nothing else, and works out which of k groups each row belongs to. Reach for it to sort things you have not named yourself, spawn points into territories, players into playstyles, tiles into biomes.

DTDAKMeans.new(k, max_iterations, num_runs) takes:
- k : how many groups to look for, 3 by default. This is yours to choose, the model will not question it
- max_iterations : how many passes one run may take before giving up, 100 by default. A run normally stops on its own, when no row changes group
- num_runs : how many times to start over from a fresh set of centres, 5 by default, keeping the run with the lowest inertia

What you get:
- predict(X) : the group each row belongs to, as an index from 0 to k-1
- fit_predict(X) : both at once, on the rows you are fitting
- inertia : the sum of the squared distances from every training row to its centre, left behind by fit()
- inertia_of(X) : the same measure for any other rows
- get_centroids() : the centres, in the unit of the training data, which is what you want to draw
- set_seed(value) and reset() : as everywhere else, a run is only repeatable once it has a seed

Distances are euclidean, so a column counted in tens of thousands would drown a column counted in units. The rows are standardised internally, the way DTDALinReg does it: you do not have to scale anything, and multiplying a column by a thousand does not change the answer.

Where the centres start decides where they end, and a poor start stays poor: it is not noise that averages out over the iterations. Both usual answers are built in. k-means++ draws the first centre among the rows, then each of the others with a weight of its squared distance to the nearest centre already chosen, so the starts spread out instead of huddling. On top of that, num_runs starts are tried and the tightest is kept. On four blobs strung along a line, eight runs beat a single one on fifty five seeds out of sixty, and never did worse.

Inertia falls as k rises whatever the grouping is worth, right down to zero when k reaches the number of rows. It cannot be read as a score on its own: it is read across several k on the same data, for where it stops falling sharply.

Example:
- var kmeans = DTDAKMeans.new(3)
- kmeans.set_seed(1)
- print("Groups: ", kmeans.fit_predict(positions))
- print("Centres: ", kmeans.get_centroids())
- for k in range(1, 6): #where does it stop falling
-  var trial = DTDAKMeans.new(k)
-  trial.set_seed(1)
-  trial.fit(positions)
-  print(k, " groups, inertia ", trial.inertia)
- kmeans.save("user://camps.json")

=== Q-Learning Model ===

Use DTDAQLearning.new() to create a new agent. This one is not trained on a dataset: it learns while playing, from the transitions you feed it. It is the model to reach for when you want an NPC that gets better at something instead of a classifier over a CSV.

DTDAQLearning.new(learning_rate, discount_factor, exploration_rate, exploration_decay, min_exploration_rate) takes:
- learning_rate : how much a single transition moves a value, 0.1 by default. Higher learns faster but is noisier
- discount_factor : how much a future reward is worth compared to an immediate one, 0.9 by default. Close to 0 the agent is greedy, close to 1 it plans ahead
- exploration_rate : epsilon, the share of moves taken at random rather than greedily, 1.0 by default so the agent starts by trying everything
- exploration_decay : what epsilon is multiplied by at the end of each episode, 0.99 by default
- min_exploration_rate : the floor epsilon never goes below, 0.01 by default, so the agent keeps a little curiosity

Epsilon is a probability: the rate and its floor are brought into [0, 1] when the agent is built, and decay_exploration() keeps epsilon in [min_exploration_rate, 1] whatever decay you give it. Nothing else is validated: a discount_factor of 1 or more diverges on a looping world, that one is on you.

The loop:
- choose_action(state, valid_actions) : epsilon-greedy, picks among the actions that are legal right now. Safe to call before anything was learned, the agent then simply explores. On a state it never met every action is worth 0, so the first of the list comes out
- learn(state, action, reward, next_state, next_actions, done) : one transition, refused with a null answer when the reward is not a real number, the cell keeping the value it had. the Bellman update Q(s, a) += lr * (reward + gamma * max Q(s', a') - Q(s, a)). Pass done = true on the last transition of an episode, a terminal state has no future to add. next_actions restricts what the agent may do next, leave it out (or pass null) to look at everything already learned about next_state
- decay_exploration() : call it at the end of an episode, epsilon goes down one notch and never below its floor
- predict(state, valid_actions) : the learned policy with no exploration at all, this is what you ship. valid_actions is optional, leave it out (or pass null) to pick among everything learned in that state. On a state the agent never met, or one whose row holds no action, it reports an error and answers null, with or without a list: it will not dress up a tie between zeros as a policy. Use choose_action() when you need a move no matter what
- get_q(state, action) : the value of a pair, 0.0 when it was never met
- set_seed(value) : fix the random draws for a reproducible run
- reset() : forget the table, put epsilon back where it started and replay the seed given to set_seed()

States and actions can be anything, they are not assumed to be contiguous integers: a Vector2i tile, a string, a dictionary key. Both are stored by their str(), which is also what a JSON object needs, so a saved agent comes back keyed exactly the same way. An action is answered back with its own type for int, float, bool, String and StringName.

What keying by str() costs, in exchange:
- two values with the same str() are the same key: the integer 1 and the string "1" share a row, and inside one state the integer 2 and the string "2" share a cell. Use one spelling per state and per action
- which pairs collide follows the number formatting of the engine, and that moves between versions: str(2.0) prints "2" up to Godot 4.3 and "2.0" from 4.4 on, so a whole float and an integer are one key on 4.3 and two different ones on 4.4. Do not rely on either behaviour
- across two states, colliding actions keep their own values but not their type: the type of an action is remembered once for the whole agent, so the last one learned decides what predict() hands back everywhere
- a float key goes through str(), which may not carry every digit of a double: depending on the engine version, 1.0/3.0 comes back exactly or a hair off. Quantize a continuous state, a position for instance, rather than rely on either
- for the same reason, a file holding float keys is tied to the engine it was written on: an agent saved on 4.3 with the state 2.0 wrote the key "2", which 4.4 spells "2.0" and would no longer find. Integer and string keys are unaffected

A saved agent carries the version of the format it was written in, and load() refuses anything else rather than reading it wrong.

Example:
- var agent = DTDAQLearning.new(0.2, 0.9, 1.0, 0.999, 0.1)
- for episode in 1000:
-  var state = start_state
-  while not done:
-   var action = agent.choose_action(state, ["left", "right"])
-   #play the action, get the reward and the next state from your game
-   agent.learn(state, action, reward, next_state, ["left", "right"], done)
-   state = next_state
-   agent.decay_exploration()
- agent.save("user://agent.json")
- print("Best move: ", agent.predict(state)) #no exploration left, the learned policy
