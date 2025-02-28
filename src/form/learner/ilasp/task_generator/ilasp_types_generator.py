import re
from collections import defaultdict

from ..ilasp_common import flatten_lists

_REGEX_PREDICATE = re.compile(
    "(?P<pred>\w+)\((?P<vars>\w+(?:,\s?\w+)*)\)"
)  # allow for multiple variable although we use only one for now


def retrieve_types(
    acc_examples,
    rej_examples,
    inc_examples,
    neg_examples,
):
    observables = set(
        flatten_lists(acc_examples, rej_examples, inc_examples, neg_examples)
    )
    types = defaultdict(list)
    for o in observables:
        m = _REGEX_PREDICATE.match(o)
        if m:
            types[m.group("pred")].append(o)
    return types
