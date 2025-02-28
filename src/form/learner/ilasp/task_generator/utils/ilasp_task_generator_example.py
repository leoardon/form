from ...ilasp_common import OBS_STR, generate_injected_statement


def generate_examples(goal_examples, deadend_examples, inc_examples, neg_examples):
    is_rejecting = len(deadend_examples) > 0

    goal_example_last_id = len(goal_examples) - 1
    dend_example_last_id = goal_example_last_id + len(deadend_examples)
    inc_example_last_id = dend_example_last_id + len(inc_examples)
    neg_example_last_id = inc_example_last_id + len(neg_examples)

    goal_example_ids = range(0, goal_example_last_id + 1)
    dend_example_ids = range(goal_example_last_id + 1, dend_example_last_id + 1)
    inc_example_ids = range(dend_example_last_id + 1, inc_example_last_id + 1)
    neg_example_ids = range(inc_example_last_id + 1, neg_example_last_id + 1)

    examples = _generate_goal_examples(goal_examples, goal_example_ids, is_rejecting)
    examples += _generate_deadend_examples(deadend_examples, dend_example_ids)
    examples += _generate_neg_examples(neg_examples, neg_example_ids, is_rejecting)
    examples += (
        _generate_incomplete_examples(inc_examples, inc_example_ids, is_rejecting)
        + "\n"
    )
    examples += (
        _generate_examples_injection(
            goal_examples, deadend_examples, inc_examples, neg_examples
        )
        + "\n"
    )
    return examples


def _generate_examples_injection(
    goal_examples, deadend_examples, inc_examples, neg_examples
):
    num_examples = (
        len(goal_examples)
        + len(deadend_examples)
        + len(inc_examples)
        + len(neg_examples)
    )
    ret = ""
    for i in range(num_examples):
        ret += generate_injected_statement(f"example_active({to_ex_id(i)}).")
        ret += "\n"
    return ret


def get_longest_example_length(
    goal_examples, deadend_examples, inc_examples, neg_examples
):
    max_goal = len(max(goal_examples, key=len)) if len(goal_examples) > 0 else 0
    max_deadend = (
        len(max(deadend_examples, key=len)) if len(deadend_examples) > 0 else 0
    )
    max_inc = len(max(inc_examples, key=len)) if len(inc_examples) > 0 else 0
    max_neg = len(max(neg_examples, key=len)) if len(neg_examples) > 0 else 0
    return max(max_goal, max_deadend, max_inc, max_neg)


def _generate_goal_examples(examples, example_ids, is_rejecting):
    example_str = ""
    for example, ex_idx in zip(examples, example_ids):
        if is_rejecting:
            example_str += f"#pos({to_ex_id(ex_idx)}, {{accept}}, {{reject}}, {{\n"
        else:
            example_str += f"#pos({to_ex_id(ex_idx)}, {{accept}}, {{}}, {{\n"
        example_str += generate_example(example)
        example_str += "}).\n\n"
    return example_str


def _generate_deadend_examples(examples, example_ids):
    example_str = ""
    for example, ex_idx in zip(examples, example_ids):
        example_str += f"#pos({to_ex_id(ex_idx)}, {{reject}}, {{accept}}, {{\n"
        example_str += generate_example(example)
        example_str += "}).\n\n"
    return example_str


def _generate_neg_examples(examples, example_ids, is_rejecting):
    example_str = ""
    for example, ex_idx in zip(examples, example_ids):
        if is_rejecting:
            example_str += f"#neg({to_ex_id(ex_idx)}, {{accept}}, {{reject}}, {{\n"
        else:
            example_str += f"#neg({to_ex_id(ex_idx)}, {{accept}}, {{}}, {{\n"
        example_str += generate_example(example)
        example_str += "}).\n\n"
    return example_str


def _generate_incomplete_examples(examples, example_ids, is_rejecting):
    example_str = ""
    for example, ex_idx in zip(examples, example_ids):
        if is_rejecting:
            example_str += f"#pos({to_ex_id(ex_idx)}, {{}}, {{accept, reject}}, {{\n"
        else:
            example_str += f"#pos({to_ex_id(ex_idx)}, {{}}, {{accept}}, {{\n"
        example_str += generate_example(example)
        example_str += "}).\n\n"
    return example_str


def generate_example(example):
    example_str = "    "
    first = True

    for i in range(0, len(example)):
        for symbol in example[i]:
            if not first:
                example_str += " "
            example_str += "%s(%s, %d)." % (OBS_STR, symbol, i)
            first = False

    if len(example) > 0:
        example_str += "\n"

    example_str += "    last(%d).\n" % (len(example) - 1)

    return example_str


def to_ex_id(ex_idx):
    return f"ex{ex_idx}"
