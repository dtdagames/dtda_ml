extends DTDATools

class_name MLTools

# === The older name === #
# The toolbox used to be called MLTools, a name generic enough to collide with the
# next addon that has one. It is DTDATools now, alongside DTDAKNN, DTDAScaler and the
# rest, and this empty subclass keeps the old name working: MLTools.new() answers
# something that carries every method DTDATools does.
#
# Two things do change, and neither can be helped.
#
# The models extend DTDATools rather than this, so "model is MLTools" is false where
# it used to be true. Ask "model is DTDATools" instead. That one at least answers
# something you can see.
#
# The other one is silent, so read it twice if you have ever extended a model. Until
# this release _predict() was the only form there was, so a subclass that overrides
# anything overrides the underscored name. The library now calls the plain names among
# themselves, and dynamic dispatch follows the name that is called: DTDAKMeans's
# fit_predict() calls predict(), so an override of _predict() is simply not reached
# there any more, and nothing says so. Measured: fit_predict() on a subclass that
# overrides _predict() answers the base grouping, and the override never runs.
#
# Move the override to predict(), and the two spellings agree again: the internal
# calls reach it because they call predict(), and a caller that still writes
# model._predict(X) reaches it too, the older name forwarding to the overridden one.
# The same goes for fit(), transform(), to_dict() and every other pair.
# === End the older name === #
