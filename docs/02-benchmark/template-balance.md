# Candidate vocabulary coverage

To describe a generated candidate rule set, run:

```sh
uv run python -m exposure.template_balance --rules outputs/templates.json --output outputs/template-balance.json
```

The input is a JSON mapping of template names to predicate lists, in the same
format accepted by `exposure.measure`. The audit groups present templates by
their declared predicate counts: `k_attr` is the number of attribute entries
and `k_rel` the number of relation entries. Duplicate predicate entries count
towards these raw complexity counts; they should be removed or rejected in a
separate rule-set validation step if they are not intended.

Each cell reports its template count and, for every supported color, shape, and
relation, a count and fraction of templates containing that value. Each value is
counted at most once per template even if multiple predicates use it. Explicit
zeroes identify vocabulary items absent from a present cell. The object-variable
histogram counts templates by their number of distinct variable names.

`missing_expected_cells` lists absent combinations in `k_attr` 1–3 and `k_rel`
0–2. Absence is not a feasibility result. The report is descriptive: equal
counts do not establish statistical balance, controlled exposure, checker
validity, or a Stage 0 pass. It does not sample scenes or train a model.
