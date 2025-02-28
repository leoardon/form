from collections import defaultdict

from ....reward_machine import RewardMachine, Rule
from ..clingo_rule_checker import ClingoChecker
from ..ilasp_common import CONNECTED_STR, N_TRANSITION_STR, PRED_SUFFIX, flatten_lists
from ..task_parser.ilasp_parser_utils import (
    parse_edge_rule,
    parse_negative_transition_rule,
)


def parse_ilasp_solutions(ilasp_learnt_filename, types):
    observables = set(flatten_lists(*types.values()))
    with open(ilasp_learnt_filename) as f:
        rm = RewardMachine()
        rm.events.update(observables)
        edges = {}
        edge_variables = defaultdict(int)
        for line in f:
            line = line.strip()
            if line.startswith(N_TRANSITION_STR):
                parsed_transition = parse_negative_transition_rule(line)
                current_edge = (
                    (parsed_transition.src, parsed_transition.dst),
                    parsed_transition.edge,
                )
                if current_edge not in edges:
                    edges[current_edge] = []
                for pos_fluent in parsed_transition.pos:
                    if pos_fluent.startswith(PRED_SUFFIX):
                        edges[current_edge].append(
                            _parse_fol(pos_fluent, edge_variables[current_edge], True)
                        )
                        edge_variables[current_edge] += 1
                    else:
                        edges[current_edge].append("~" + pos_fluent)
                for neg_fluent in parsed_transition.neg:
                    if neg_fluent.startswith(PRED_SUFFIX):
                        edges[current_edge].append(
                            _parse_fol(neg_fluent, edge_variables[current_edge], False)
                        )
                        edge_variables[current_edge] += 1
                    else:
                        edges[current_edge].append(neg_fluent)
            elif line.startswith(CONNECTED_STR):
                parsed_edge = parse_edge_rule(line)
                current_edge = ((parsed_edge.src, parsed_edge.dst), parsed_edge.edge)
                if current_edge not in edges:
                    edges[current_edge] = []

        for edge, rule in edges.items():
            from_state, to_state = edge[0]
            rm.add_states([from_state, to_state])
            rm.add_transition(
                from_state,
                to_state,
                Rule(
                    tuple(rule),
                    ClingoChecker(tuple(rule), types),
                ),
            )

        return rm


def _parse_fol(fluent, counter, positive_transition):
    pred = fluent.replace(PRED_SUFFIX, "")[:-1]
    quantifier, pred = pred.split("_", 1)
    if quantifier == "a":
        q = "∀" if not positive_transition else "∃"
        neg = "" if not positive_transition else "~"
    elif quantifier == "e":
        q = "∃" if not positive_transition else "∀"
        neg = "" if not positive_transition else "~"
    return f"{q}X{counter}, {neg}{pred}(X{counter})"
