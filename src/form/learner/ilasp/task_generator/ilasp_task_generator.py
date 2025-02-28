import os

from ..ilasp_common import flatten_lists, generate_types_statements
from .utils.ilasp_task_generator_example import generate_examples
from .utils.ilasp_task_generator_hypothesis import get_hypothesis_space
from .utils.ilasp_task_generator_state import generate_state_statements
from .utils.ilasp_task_generator_symmetry_breaking import (
    generate_symmetry_breaking_statements,
)
from .utils.ilasp_task_generator_transition import (
    generate_state_at_timestep_statements,
    generate_timestep_statements,
    generate_transition_statements,
)


def generate_ilasp_task(
    num_states,
    accepting_state,
    rejecting_state,
    acc_examples,
    rej_examples,
    inc_examples,
    neg_examples,
    types,
    output_folder,
    output_filename,
    symmetry_breaking_method,
    max_disj_size,
    learn_acyclic,
    use_compressed_traces,
    avoid_learning_only_negative,
    prioritize_optimal_solutions,
    binary_folder_name=None,
):
    # statements will not be generated for the rejecting state if there are not deadend examples
    if len(rej_examples) == 0:
        rejecting_state = None
    # it is possible to have only negative examples. there should not be an accepting state in that case
    if len(acc_examples) == 0:
        accepting_state = None

    observables = set(
        flatten_lists(acc_examples, rej_examples, inc_examples, neg_examples)
    )

    with open(os.path.join(output_folder, output_filename), "w") as f:
        task = _generate_ilasp_task_str(
            num_states,
            accepting_state,
            rejecting_state,
            observables,
            acc_examples,
            rej_examples,
            inc_examples,
            neg_examples,
            output_folder,
            symmetry_breaking_method,
            max_disj_size,
            learn_acyclic,
            use_compressed_traces,
            avoid_learning_only_negative,
            prioritize_optimal_solutions,
            binary_folder_name,
            types,
        )
        f.write(task)


def _generate_ilasp_task_str(
    num_states,
    accepting_state,
    rejecting_state,
    observables,
    acc_examples,
    rej_examples,
    inc_examples,
    neg_examples,
    output_folder,
    symmetry_breaking_method,
    max_disj_size,
    learn_acyclic,
    use_compressed_traces,
    avoid_learning_only_negative,
    prioritize_optimal_solutions,
    binary_folder_name,
    types,
):
    task = generate_state_statements(num_states, accepting_state, rejecting_state)
    task += generate_timestep_statements(
        acc_examples, rej_examples, inc_examples, neg_examples
    )
    task += _generate_edge_indices_facts(max_disj_size)
    task += generate_state_at_timestep_statements(
        num_states, accepting_state, rejecting_state
    )
    task += generate_types_statements(types)
    task += generate_transition_statements(
        learn_acyclic,
        use_compressed_traces,
        avoid_learning_only_negative,
        prioritize_optimal_solutions,
        types,
    )
    task += get_hypothesis_space(
        num_states,
        accepting_state,
        rejecting_state,
        observables,
        output_folder,
        symmetry_breaking_method,
        max_disj_size,
        learn_acyclic,
        binary_folder_name,
        types,
    )

    if symmetry_breaking_method is not None:
        task += generate_symmetry_breaking_statements(
            num_states,
            accepting_state,
            rejecting_state,
            observables,
            symmetry_breaking_method,
            max_disj_size,
            learn_acyclic,
            types,
        )
    task += generate_examples(acc_examples, rej_examples, inc_examples, neg_examples)

    return task


def _generate_edge_indices_facts(max_disj_size):
    return "edge_id(1..%d).\n\n" % max_disj_size
