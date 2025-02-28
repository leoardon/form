import itertools
import logging
import os

from .task_generator import generate_ilasp_task, retrieve_types
from .task_parser import parse_ilasp_solutions
from .task_solver import solve_ilasp_task

LOGGER = logging.getLogger(__name__)


class ILASPLearner:
    def __init__(
        self,
        agent_id,
        init_rm_num_states=None,
        max_rm_num_states=None,
        wait_for_pos_only=True,
        compressed_trace=True,
        learn_acyclic=False,
        learn_first_order=True,
        max_disjunctions=None
    ):
        self.agent_id = agent_id
        self._log_folder = None
        self.rm_learning_counter = 0

        self.init_rm_num_states = init_rm_num_states
        self.max_rm_num_states = max_rm_num_states
        self.rm_num_states = max_rm_num_states if not init_rm_num_states else init_rm_num_states
        self.wait_for_pos_only = wait_for_pos_only
        self.compressed_trace = compressed_trace
        self.learn_acyclic = learn_acyclic
        self.learn_first_order = learn_first_order
        self.max_disjunctions = max_disjunctions

        self._previous_positive_examples = None
        self._previous_deadend_examples = None
        self._previous_incomplete_examples = None
        self._previous_rm_num_states = None

    @property
    def log_folder(self):
        if self._log_folder is None:
            raise RuntimeError("log_folder should be set")
        return self._log_folder
    
    def set_log_folder(self, folder):
        if not os.path.exists(folder):
            os.mkdir(folder)
        self._log_folder = folder

    def learn(self, rm, positive_examples, deadend_examples, incomplete_examples):
        return self._update_reward_machine(
            rm,
            self.process_examples(positive_examples),
            self.process_examples(deadend_examples),
            self.process_examples(incomplete_examples),
        )

    def process_examples(self, examples):
        return sorted(set(examples), key=len)[:100]

    # We assume that the set of examples is strictly increasing. So, length checking is sufficient to check for
    # equality.
    def _have_changed(self, positive_examples, deadend_examples, incomplete_examples):
        if (
            self._previous_positive_examples is None
            or set(positive_examples) != self._previous_positive_examples
        ):
            return True

        if (
            self._previous_deadend_examples is None
            or set(deadend_examples) != self._previous_deadend_examples
        ):
            return True

        if (
            self._previous_incomplete_examples is None
            or set(incomplete_examples) != self._previous_incomplete_examples
        ):
            return True

        return False

    def _update_reward_machine(
        self,
        rm,
        positive_examples,
        deadend_examples,
        incomplete_examples,
        rm_num_states=None,
    ):
        LOGGER.debug(f"[{self.agent_id}]`_update_reward_machine`")

        if self.wait_for_pos_only:
            if not positive_examples:
                LOGGER.debug(f"[{self.agent_id}] No positive examples")
                return
        else:
            if not positive_examples and not deadend_examples:
                LOGGER.debug(f"[{self.agent_id}] No positive and no deadend examples")
                return

        is_new_learning = rm_num_states is None

        rm_num_states = (
            rm_num_states
            or min(
                self.rm_num_states - (1 if not deadend_examples else 2),
                min(len(t) for t in itertools.chain(positive_examples))
            ) + (1 if not deadend_examples else 2)
        )

        if self.max_rm_num_states and rm_num_states - (1 if not deadend_examples else 2) > self.max_rm_num_states:
            LOGGER.debug(f"[{self.agent_id}] Trying to learn with more states than max_rm_num_states {self.max_rm_num_states}")
            return

        if (
            not self._have_changed(
                positive_examples, deadend_examples, incomplete_examples
            )
            and rm_num_states == self._previous_rm_num_states
        ):
            LOGGER.debug(f"[{self.agent_id}] Examples haven't changed")
            return
        else:
            self._previous_positive_examples = set(positive_examples)
            self._previous_deadend_examples = set(deadend_examples)
            self._previous_incomplete_examples = set(incomplete_examples)
            self._previous_rm_num_states = rm_num_states

        LOGGER.debug(f"[{self.agent_id}] num_state: {rm_num_states}")

        if is_new_learning:
            self.rm_learning_counter += 1

        LOGGER.debug(
            f"[{self.agent_id}] generating task {self.rm_learning_counter} with {rm_num_states}: start"
        )

        ilasp_task_filename = f"task_{self.rm_learning_counter}_{rm_num_states}"
        ilasp_solution_filename = f"solution_{self.rm_learning_counter}_{rm_num_states}"

        examples = (
            sorted(positive_examples, key=len),
            sorted(deadend_examples, key=len),
            sorted(incomplete_examples, key=len),
            [],
        )
        types = retrieve_types(*examples) if self.learn_first_order else {}
        self._generate_ilasp_task(
            ilasp_task_filename,
            *examples,
            types,
            rm_num_states,
        )
        LOGGER.debug(
            f"[{self.agent_id}] generating task {self.rm_learning_counter} with {rm_num_states}: done"
        )
        LOGGER.debug(
            f"[{self.agent_id}] solving task {self.rm_learning_counter} with {rm_num_states}: start"
        )
        solver_success = self._solve_ilasp_task(
            ilasp_task_filename, ilasp_solution_filename
        )
        LOGGER.debug(
            f"[{self.agent_id}] solving task {self.rm_learning_counter} with {rm_num_states}: done"
        )
        if solver_success:
            ilasp_solution_filename = os.path.join(
                self.log_folder, ilasp_solution_filename
            )
            candidate_rm = parse_ilasp_solutions(ilasp_solution_filename, types)

            if candidate_rm.states:
                self.rm_num_states = self.init_rm_num_states

                candidate_rm.set_u0("u0")
                if positive_examples:
                    candidate_rm.set_uacc("u_acc")
                if deadend_examples:
                    candidate_rm.set_urej("u_rej")

                if candidate_rm != rm:
                    rm_plot_filename = os.path.join(
                        self.log_folder,
                        f"plot_{self.rm_learning_counter}_{rm_num_states}",
                    )
                    candidate_rm.plot(rm_plot_filename)
                    rm_plot_filename_prev = os.path.join(
                        self.log_folder, f"plot_{self.rm_learning_counter}_prev"
                    )
                    rm.plot(rm_plot_filename_prev)
                    return candidate_rm
            else:
                LOGGER.debug(f"[{self.agent_id}] ILASP task unsolvable")
                if self.init_rm_num_states:
                    self.rm_num_states += 1
                return self._update_reward_machine(
                    rm,
                    positive_examples,
                    deadend_examples,
                    incomplete_examples,
                    rm_num_states=(rm_num_states + 1),
                )
        else:
            # raise RuntimeError(
            #     "Error: Couldn't find an automaton within the specified timeout!"
            # )
            return

    def _generate_ilasp_task(
        self,
        ilasp_task_filename,
        positive_examples,
        deadend_examples,
        incomplete_examples,
        negative_examples,
        types,
        rm_num_states,
    ):
        if not self.max_disjunctions:
            if not self.learn_first_order:
                examples = (
                    positive_examples,
                    deadend_examples,
                    incomplete_examples,
                    [],
                )
                types_ = retrieve_types(*examples)
                max_disjunctions = max([len(v) for v in types_.values()])
            else:
                max_disjunctions = 1
        else:
            max_disjunctions = self.max_disjunctions
        # the sets of examples are sorted to make sure that ILASP produces the same solution for the same sets (ILASP
        # can produce different hypothesis for the same set of examples but given in different order)
        generate_ilasp_task(
            rm_num_states,
            "u_acc",
            "u_rej",
            positive_examples,
            deadend_examples,
            incomplete_examples,
            negative_examples,
            types,
            self.log_folder,
            ilasp_task_filename,
            "bfs-alternative",  # symmetry_breaking_method
            max_disjunctions,  # max_disjunction_size
            self.learn_acyclic,  # learn_acyclic_graph
            self.compressed_trace,  # use_compressed_traces
            True,  # avoid_learning_only_negative
            False,  # prioritize_optimal_solutions
            None # bin directory (ILASP is on PATH)
        )

    def _solve_ilasp_task(self, ilasp_task_filename, ilasp_solution_filename):
        ilasp_task_filename = os.path.join(self.log_folder, ilasp_task_filename)

        ilasp_solution_filename = os.path.join(self.log_folder, ilasp_solution_filename)
        return solve_ilasp_task(
            ilasp_task_filename,
            ilasp_solution_filename,
            timeout=3600,
            version="2",
            max_body_literals=1,
            binary_folder_name=None,
            compute_minimal=True,
        )
