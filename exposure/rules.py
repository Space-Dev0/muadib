"""Evaluate conjunctions on annotated scenes with distinct object bindings.

Coordinates follow image convention: x increases rightward and y downward.
Relations compare object centers. Ties do not satisfy either relation.
"""

from collections import defaultdict
from dataclasses import dataclass
from itertools import permutations
from typing import Literal, TypeAlias


@dataclass(frozen=True)
class SceneObject:
    color: str
    shape: str
    x: float
    y: float


@dataclass(frozen=True)
class Attribute:
    variable: str
    field: Literal["color", "shape"]
    value: str

    def __post_init__(self) -> None:
        if self.field not in {"color", "shape"}:
            raise ValueError(f"Unsupported attribute: {self.field}")


@dataclass(frozen=True)
class Relation:
    subject: str
    kind: Literal["left_of", "above"]
    object: str

    def __post_init__(self) -> None:
        if self.kind not in {"left_of", "above"}:
            raise ValueError(f"Unsupported relation: {self.kind}")
        if self.subject == self.object:
            raise ValueError("A relation must refer to two distinct variables")


Predicate: TypeAlias = Attribute | Relation


def _has_cycle(edges: list[tuple[str, str]]) -> bool:
    """Return whether a strict directed graph contains a cycle."""
    graph: dict[str, set[str]] = defaultdict(set)
    nodes: set[str] = set()
    for source, target in edges:
        graph[source].add(target)
        nodes.update((source, target))
    state: dict[str, int] = {}

    def visit(node: str) -> bool:
        state[node] = 1
        for target in graph[node]:
            if state.get(target) == 1:
                return True
            if state.get(target, 0) == 0 and visit(target):
                return True
        state[node] = 2
        return False

    return any(state.get(node, 0) == 0 and visit(node) for node in sorted(nodes))


def validate_rule(predicates: tuple[Predicate, ...]) -> tuple[Predicate, ...]:
    """Reject rules that are redundant or structurally impossible.

    Duplicate identical predicates double the apparent complexity without
    changing semantics.  Two different values for the same attribute on the
    same variable are unsatisfiable.  A cycle in either strict order relation
    is also unsatisfiable.  Rejecting these at parse time prevents them from
    entering measurement, design, or model-fitting code as if they were valid.
    """
    if not predicates:
        raise ValueError("A rule must contain at least one predicate")
    seen: set[Predicate] = set()
    attributes: dict[tuple[str, str], str] = {}
    edges: dict[str, list[tuple[str, str]]] = {"left_of": [], "above": []}
    for predicate in predicates:
        if predicate in seen:
            raise ValueError(f"Duplicate predicate: {predicate!r}")
        seen.add(predicate)
        if isinstance(predicate, Attribute):
            key = (predicate.variable, predicate.field)
            if key in attributes and attributes[key] != predicate.value:
                raise ValueError(
                    f"Contradictory {predicate.field} values for variable "
                    f"{predicate.variable!r}: {attributes[key]!r} and {predicate.value!r}")
            attributes[key] = predicate.value
        elif isinstance(predicate, Relation):
            edges[predicate.kind].append((predicate.subject, predicate.object))
        else:
            raise TypeError(f"Unsupported predicate: {predicate!r}")
    for kind, pairs in edges.items():
        if _has_cycle(pairs):
            raise ValueError(f"Cyclic {kind} constraints make the rule unsatisfiable")
    return tuple(predicates)


def satisfies(scene: tuple[SceneObject, ...], predicates: tuple[Predicate, ...]) -> bool:
    """Return whether some assignment of distinct objects satisfies every predicate.

    Evaluating a subset independently existentially quantifies only the variables
    present in that subset. Shared variable names retain their binding within it.
    """
    if not predicates:
        raise ValueError("A rule must contain at least one predicate")

    variables: set[str] = set()
    for predicate in predicates:
        if isinstance(predicate, Attribute):
            variables.add(predicate.variable)
        elif isinstance(predicate, Relation):
            variables.update((predicate.subject, predicate.object))
        else:
            raise TypeError(f"Unsupported predicate: {predicate!r}")

    names = sorted(variables)
    for objects in permutations(scene, len(names)):
        assignment = dict(zip(names, objects, strict=True))
        if all(_matches(predicate, assignment) for predicate in predicates):
            return True
    return False


def _matches(predicate: Predicate, assignment: dict[str, SceneObject]) -> bool:
    if isinstance(predicate, Attribute):
        return getattr(assignment[predicate.variable], predicate.field) == predicate.value
    subject = assignment[predicate.subject]
    target = assignment[predicate.object]
    if predicate.kind == "left_of":
        return subject.x < target.x
    return subject.y < target.y
