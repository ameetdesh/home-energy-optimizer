# Origin

This grew out of an unpublished Pyodide proof of concept that ran decomposed
per-device dynamic programs in the browser.

What carried over: the approach — a 1-D DP per device, coordinated by a price
signal — and the observation that a DP yields a value function, which is what
makes marginal prices and counterfactuals cheap.

What changed on the way here:

- extracted from the browser page into a library with no DOM or Pyodide
  dependency, and given a test suite (the POC had none; correctness was judged
  by looking at a chart)
- three defects fixed that the missing tests had hidden, one of which disabled
  the battery entirely
- comfort priced in currency rather than an arbitrary weight
- real forecast inputs, grid limits, PV curtailment, multiple batteries
- the value function surfaced deliberately as prices, rather than used only
  internally
- validated against exact references: a continuous LP, an exact joint DP, and a
  mixed-integer solver over the same model

`docs/NOTES.md` records the measurements, including where the approach is worse
than an exact solve and by how much.

## Where the emphasis is now

The value function turned out to be one convenience among several. The project
is now about cooperation: devices with different optimisers - dynamic, linear or
quadratic programmes, or someone else's black box - answering one interface,
coordinated into one plan for the house by Dantzig–Wolfe or ADMM, with the
saving shared fairly among them.
