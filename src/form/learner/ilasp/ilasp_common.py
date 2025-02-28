import itertools

N_TRANSITION_STR = "n_phi"
CONNECTED_STR = "ed"
OBS_STR = "obs"
PRED_STR = "pred"
PRED_SUFFIX = f"{PRED_STR}("


def generate_injected_statements(stmts):
    return "\n".join([generate_injected_statement(stmt) for stmt in stmts]) + "\n"


def generate_injected_block(stmts):
    return generate_injected_statement("\n\t" + "\n\t".join(stmts) + "\n") + "\n"


def generate_injected_statement(stmt):
    return '#inject("' + stmt + '").'


def generate_types_statements(types):
    stmt = "\n%%%%%%%%%%%%%%%%%%%%%%"
    for typ, symbols in types.items():
        stmt += "\n"
        for s in symbols:
            stmt += f"{typ}({s}).\n"
        stmt += "\n"

        # existential
        stmt += f"e_{typ}_at(T) :- obs(O, T), {typ}(O).\n"

        # universal
        stmt += f"aux_{typ}_seen_at(O, T) :- obs(O, T2), {typ}(O), T >= T2, st(T, X), st(T2, X), step(T), step(T2), state(X).\n"
        stmt += f"not_all_{typ}_seen_at(T) :- not aux_{typ}_seen_at(O, T), {typ}(O), step(T).\n"
        stmt += f"all_{typ}_seen_at(T) :- not not_all_{typ}_seen_at(T), step(T).\n"
        stmt += (
            f"aux_a_{typ}_at(T) :- all_{typ}_seen_at(T2), T2<T, step(T2), step(T).\n"
        )
        stmt += (
            f"a_{typ}_at(T) :- all_{typ}_seen_at(T), not aux_a_{typ}_at(T), step(T).\n"
        )

    stmt += "%%%%%%%%%%%%%%%%%%%%%%\n\n"
    return stmt


def flatten_lists(*args):
    if not args:
        return []
    while isinstance(args[0], (list, tuple)):
        args = list(itertools.chain(*args))
    return args
