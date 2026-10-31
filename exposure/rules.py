"""Evaluate conjunctions on annotated scenes with distinct object bindings.

Coordinates follow image convention: x increases rightward and y downward.
Relations compare object centers. Ties do not satisfy either relation.
"""

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
