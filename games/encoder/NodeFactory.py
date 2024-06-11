from pymimir import Atom, Literal, Object, Predicate
from functools import singledispatchmethod

Node = str
class NodeFactory:
    
    def __init__(self, negation_prefix: str = "~", goal_suffix: str = "_g"):
        self.negation_prefix = negation_prefix
        self.goal_suffix = goal_suffix

    def __eq__(self, other):
        return (
            isinstance(other, NodeFactory)
            and self.negation_prefix == other.negation_prefix
            and self.goal_suffix == other.goal_suffix
        )

    @singledispatchmethod
    def __call__(self, item, *args, **kwargs) -> Node | None:
        raise NotImplementedError(
            "__call__ is not implemented for type {}".format(type(item))
        )

    @__call__.register
    def atom_node(
        self,
        atom: Atom,
        pos: int | None = None,
        *args,
        as_predicate: bool = False,
        **kwargs,
    ) -> Node | None:
        if as_predicate:
            return self(atom.predicate, is_goal=False, is_negated=False)
        return f"{atom.get_name()}:{pos}"

    @__call__.register
    def predicate_node(
        self,
        predicate: Predicate,
        *,
        is_goal: bool = False,
        is_negated: bool = False,
        **kwargs,
    ) -> Node | None:
        prefix = self.negation_prefix if is_negated else ""
        suffix = self.goal_suffix if is_goal else ""
        return f"{prefix}{predicate.name}{suffix}"

    @__call__.register
    def literal_node(
        self,
        literal: Literal,
        pos: int | None = None,
        *,
        as_predicate: bool = False,
        **kwargs,
    ) -> Node | None:
        if as_predicate:
            # by default, we assume that literals are goal atoms
            return self(
                literal.atom.predicate,
                is_goal=kwargs.get("is_goal", True),
                is_negated=literal.negated,
            )
        prefix = self.negation_prefix if literal.negated else ""
        pos_string = f":{pos}" if pos is not None else ""
        return f"{prefix}{literal.atom.get_name()}{self.goal_suffix}{pos_string}"

    @__call__.register
    def object_node(self, obj: Object, *args, **kwargs) -> Node | None:
        return obj.name

    @__call__.register
    def none_node(self, none: None, *args, **kwargs) -> Node | None:
        return None

    def create_node(self, obj, obj_type=None, as_predicate=False, pos=None):
        if as_predicate:
            return f"predicate_{obj}"
        if pos is not None:
            return f"{obj_type}_{obj}_{pos}"
        return f"{obj_type}_{obj}"

    def create_object_node(self, obj):
        return f"object_{id(obj)}"

    def create_predicate_node(self, predicate):
        return f"predicate_{predicate}"

    def create_distance_atom_node(self, pacman, ghost):
        return f"distance_{id(pacman)}_{id(ghost)}"
