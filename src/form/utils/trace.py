from collections import OrderedDict
from itertools import groupby


class TraceTracker:
    def __init__(self, compressed: bool = True) -> None:
        self.compressed = compressed
        self.trace = []
        self.state_trace = []
        self.obs = []

        self._current_state = None

        self._hash_state_mapping = OrderedDict()

    def reset(self):
        self.trace.clear()
        self.state_trace.clear()
        self.obs.clear()

        self._current_state = None

    def update(self, labels, obs, state=None):
        prev_len_labels_seq = len(self.labels_sequence)

        if self._current_state is None or self._current_state != state:
            self._current_state = state
            self.state_trace.clear()

        self.trace.append(self._process_label(labels))
        self.state_trace.append(self._process_label(labels))
        self.obs.append(self._process_obs(obs))

        if prev_len_labels_seq == len(self.labels_sequence):
            return tuple()
        return labels

    def _process_label(self, labels):
        return labels or tuple()

    def _process_obs(self, obs):
        state_hash = hash(str(obs))
        if state_hash not in self._hash_state_mapping:
            self._hash_state_mapping[state_hash] = obs
        return list(self._hash_state_mapping.keys()).index(state_hash) + 1

    @property
    def _labels_sequence(self):
        return tuple(tuple(es) for es in self.trace)

    @property
    def _state_labels_sequence(self):
        return tuple(tuple(es) for es in self.state_trace)

    @property
    def _labels_sequence_no_empty(self):
        return tuple(tuple(es) for es in self.trace if es)

    @property
    def _state_labels_sequence_no_empty(self):
        return tuple(tuple(es) for es in self.state_trace if es)

    @property
    def _compressed_labels_sequence(self):
        return tuple(i[0] for i in groupby(self._labels_sequence_no_empty or tuple()))

    @property
    def _compressed_state_labels_sequence(self):
        return tuple(
            i[0] for i in groupby(self._state_labels_sequence_no_empty or tuple())
        )

    @property
    def labels_sequence(self):
        if self.compressed:
            return self._compressed_labels_sequence
        else:
            return self._labels_sequence

    @property
    def _flatten_labels_sequence(self):
        return tuple(e for es in self.trace for e in es)

    @property
    def _compressed_flatten_labels_sequence(self):
        return tuple(i[0] for i in groupby(self._flatten_labels_sequence or tuple()))

    @property
    def flatten_labels_sequence(self):
        if self.compressed:
            return self._compressed_flatten_labels_sequence
        else:
            return self._flatten_labels_sequence

    @property
    def sequence(self):
        return tuple((l, o) for l, o in zip(self.trace, self.obs) if l)

    @property
    def state_labels_sequence(self):
        if self.compressed:
            return self._compressed_state_labels_sequence
        else:
            return self._state_labels_sequence
