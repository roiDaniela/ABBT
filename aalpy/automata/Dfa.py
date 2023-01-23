from aalpy.base import AutomatonState, DeterministicAutomaton


class DfaState(AutomatonState):
    """
    Single state of a deterministic finite automaton.
    """

    def __init__(self, state_id, is_accepting=False):
        super().__init__(state_id)
        self.parent = dict()
        self.is_accepting = is_accepting


class Dfa(DeterministicAutomaton):
    """
    Deterministic finite automaton.
    """

    def __init__(self, initial_state: DfaState, states):
        super().__init__(initial_state, states)

    def cross_product(self, D2, accept_method):
        """A generalized cross-product constructor over two DFAs.
        The third argument is a binary boolean function f; a state (q1, q2) in the final
        DFA accepts if f(A[q1],A[q2]), where A indicates the acceptance-value of the state.
        """
        assert (self.get_input_alphabet() == D2.get_input_alphabet())

        alphabet = self.get_input_alphabet()

        # assert both automata are complete
        assert (all([len(s.transitions.keys())==len(alphabet) for s in self.states]))
        assert (all([len(s.transitions.keys())==len(alphabet) for s in D2.states]))

        dfa_state_setup = {}

        s1 = self.initial_state
        s2 = D2.initial_state
        name = s1.state_id + s2.state_id
        a1 = s1.is_accepting
        a2 = s2.is_accepting
        acc = accept_method(a1, a2)
        actions_s = {}
        for action in alphabet:
            actions_s[action] = s1.transitions[action].state_id + s2.transitions[action].state_id
        dfa_state_setup[name] = (acc, actions_s)

        for s1 in self.states:
            for s2 in D2.states:
                if not (s1 == self.initial_state and s2 == D2.initial_state):
                    name = s1.state_id + s2.state_id
                    a1 = s1.is_accepting
                    a2 = s2.is_accepting
                    acc = accept_method(a1, a2)
                    actions_s = {}
                    for action in alphabet:
                        actions_s[action] = s1.transitions[action].state_id + s2.transitions[action].state_id
                    dfa_state_setup[name] = (acc, actions_s)

        return dfa_state_setup

    def intersection_setup(self, D2):
        """Constructs an unminimized DFA recognizing the intersection of the languages of two given DFAs."""
        f = bool.__and__
        return self.cross_product(D2, f)

    def union_setup(D1, D2):
        """Constructs an unminimized DFA recognizing the union of the languages of two given DFAs."""
        f = bool.__or__
        return self.cross_product(D2, f)

    def symmetric_difference_setup(self, D2):
        """Constructs an unminimized DFA recognizing the symmetric difference of the languages of two given DFAs."""
        f = bool.__xor__
        return self.cross_product(D2, f)

    def step(self, letter):
        """
        Args:

            letter: single input that is looked up in the transition table of the DfaState

        Returns:

            True if the reached state is an accepting state, False otherwise
        """
        if letter is not None:
            self.current_state = self.current_state.transitions[letter]
        return self.current_state.is_accepting

    def compute_characterization_set(self, char_set_init=None, online_suffix_closure=True, split_all_blocks=True,
                                     return_same_states=False, raise_warning=True):
        return super(Dfa, self).compute_characterization_set(char_set_init if char_set_init else [()],
                                                             online_suffix_closure, split_all_blocks,
                                                             return_same_states, raise_warning)

    def is_minimal(self):
        return self.compute_characterization_set(raise_warning=False) is not None

    def compute_output_seq(self, state, sequence):
        if not sequence:
            return [state.is_accepting]
        return super(Dfa, self).compute_output_seq(state, sequence)

    def to_state_setup(self):
        state_setup_dict = {}

        # ensure prefixes are computed
        self.compute_prefixes()

        sorted_states = sorted(self.states, key=lambda x: len(x.prefix))
        for s in sorted_states:
            state_setup_dict[s.state_id] = (s.is_accepting, {k: v.state_id for k, v in s.transitions.items()})

        return state_setup_dict

