# Independent-object exposure baseline

`exposure.object_exposure_null` implements a mathematical reference for a same-object conjunction under explicit assumptions: iid objects, mutually independent attribute axes within each object, and a supplied object-count distribution independent of attributes. It does not assume that the existing renderer meets these assumptions and does not replace its empirical measurements.

For an object-level event probability `p`, a scene with fixed `K` has `P(at least one) = 1-(1-p)^K`. Under the stated independence assumptions, the same-object joint probability is `prod(p_i)`. If `K` varies, the program averages the scene probability over the supplied distribution **before** taking the product of scene-level marginals. This distinction matters because a shared random object count induces dependence between scene-level events even when attributes are independent conditional on count.

```sh
uv run --no-sync python -m exposure.object_exposure_null \
  --probabilities 0.5 0.5 --counts '{"3":1}' --observed-joint 0.578125
```

The fixed-K example gives scene marginals 0.875, independent-object scene joint 0.578125 and scene-level log lift about -0.2809, despite independent object-level attributes. The optional `observed_joint` returns `log(observed / baseline)` when both values are positive; it is meaningful only when the same scene population, object-count distribution and bound event are used. Zeros yield null log ratios rather than invented continuity corrections. For K=1 the null lift is zero; with equal 0.5 attribute probabilities it becomes more negative by K=3, then approaches zero as K grows. Thus it is not monotonically decreasing in K.

Unit tests include a fixed example, variable-K Monte Carlo verification, boundary cases and input validation. A realistic renderer null must additionally accommodate its grid, conditional scene selection, variable binding, attribute dependence and finite-sample uncertainty. Neither a corrected baseline nor a model outcome has been validated on generated images here.
