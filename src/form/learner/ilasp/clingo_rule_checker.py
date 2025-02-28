import itertools
import textwrap
from dataclasses import dataclass, field
from typing import List, Literal

from .ilasp_common import generate_types_statements
from .task_generator.utils.ilasp_task_generator_example import generate_example

@dataclass
class _BaseRule:
    is_positive: bool
    rule_type: Literal["obs", "fol"] = field(init=False)

    def pos_neg_asp(self):
        return "" if self.is_positive else "not "

@dataclass
class _ObsRule(_BaseRule):
    obs: str

    def __post_init__(self):
        self.rule_type = "obs"

    def to_asp(self):
        return f"{self.pos_neg_asp()}obs({self.obs}, T)"

@dataclass
class _FOLRule(_BaseRule):
    pred: str
    quantifier: str

    def __post_init__(self):
        self.rule_type = "fol"

    def to_asp(self):
        return f"{self.pos_neg_asp()}{self.quantifier}_{self.pred}_at(T)"

class ClingoChecker:

    def __init__(self, atoms, types):
        self.rules = self._parse_rules(atoms)
        self.types = self._filter_types(types, self.rules)
        self.template = self._build_template(self.rules, self.types)

    @classmethod
    def _parse_rules(cls, atoms) -> List[_BaseRule]:
        return [cls._parse_rule(a) for a in atoms]
    
    @classmethod
    def _parse_rule(cls, r):
        if "∀" in r or "∃" in r:
            return cls._parse_fol_rule(r)
        else:
            return cls._parse_obs_rule(r)

    @staticmethod
    def _parse_obs_rule(r):
        is_positive = "~" not in r
        obs = r.strip("~")
        return _ObsRule(is_positive=is_positive, obs=obs)

    @staticmethod
    def _parse_fol_rule(r):
        is_positive = "~" not in r
        quantifier = ""
        if "∀" in r:
            quantifier = "a" if is_positive else "e"
        elif "∃" in r:
            quantifier = "e" if is_positive else "a"
        pred = r.split(",")[1].split("(")[0].strip().strip("~")
        return _FOLRule(is_positive=is_positive, pred=pred, quantifier=quantifier)

    @staticmethod
    def _filter_types(types, rules):
        return {
            p: obs 
            for p, obs in types.items() 
            if p in [r.pred for r in rules if isinstance(r, _FOLRule)]
        }

    @classmethod
    def _build_template(cls, rules, types):
        template = generate_types_statements(types)
        template += "#defined obs/2.\n\n"
        template += ":- not aux_pred.\naux_pred :- "
        template += ", ".join([r.to_asp() for r in rules] + ["step(T)"])
        template += ".\n\n"
        return template

    def _ctrl(self):
        from clingo import Control

        ctl = Control()
        ctl.add(self.template)
        return ctl

    def __call__(self, observations, trace) -> bool:
        ctl = self._ctrl()
        
        trace = [
            tuple(
                l for l in step 
                if l in itertools.chain.from_iterable(
                    self.types.values()
                )
            )
            for step in trace[:-1]
        ] + [observations]

        ctl.add(textwrap.dedent(generate_example(trace)))
        ctl.add(f"step(0..{len(trace)-1}).")
        ctl.add(f"st(0..{len(trace)-1}, u0). state(u0).")
        ctl.ground()
        return ctl.solve().satisfiable
