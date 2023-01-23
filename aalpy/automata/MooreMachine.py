from aalpy.base import AutomatonState, DeterministicAutomaton

class MooreState(AutomatonState):
    """
    Single state of a Moore machine. Each state has an output value.
    """

    def __init__(self, state_id, output=None):
        super().__init__(state_id)
        self.output = output


class MooreMachine(DeterministicAutomaton):

    def __init__(self, initial_state: AutomatonState, states: list):
        super().__init__(initial_state, states)

    def step(self, letter):
        """
        In Moore machines outputs depend on the current state.

        Args:

            letter: single input that is looked up in the transition function leading to a new state

        Returns:

            the output of the reached state

        """
        if letter is not None:
            self.current_state = self.current_state.transitions[letter]
        return self.current_state.output

    def compute_characterization_set(self, char_set_init=None, online_suffix_closure=True, split_all_blocks=True,
                                     return_same_states=False, raise_warning=True):
        return super(MooreMachine, self).compute_characterization_set(char_set_init if char_set_init else [()],
                                                                      online_suffix_closure, split_all_blocks,
                                                                      return_same_states, raise_warning)

    def convert_2_complete_dfa_setup(self, kernel_states=None, alphabet=None):
        dfa_state_setup = {}

        if alphabet is None:
            alphabet = self.get_input_alphabet()
        sink_state_name = 'sinki'

        s = self.initial_state
        name = s.state_id
        actions_s = {}
        for action in alphabet:
            if action in s.transitions:
                actions_s[action] = s.transitions[action].state_id
            else:
                actions_s[action] = sink_state_name
        dfa_state_setup[name] = (True, actions_s)

        states = self.states if kernel_states is None else kernel_states
        for s in states:
            if s != self.initial_state:
                name = s.state_id
                actions_s = {}
                for action in alphabet:
                    if action in s.transitions:
                        actions_s[action] = s.transitions[action].state_id if kernel_states is None or s.transitions[action] in states else sink_state_name
                    else:
                        actions_s[action] = sink_state_name
                dfa_state_setup[name] = (True, actions_s)

        dfa_state_setup[sink_state_name] = (False, {a:sink_state_name for a in alphabet})

        for k,v in dfa_state_setup.items():
            for k1, v1 in v[1].items():
                if v1 not in dfa_state_setup.keys():
                    dfa_state_setup[k][1][k1] = sink_state_name
        return dfa_state_setup

    def is_minimal(self):
        return self.compute_characterization_set(raise_warning=False) is not None

    def compute_output_seq(self, state, sequence):
        if not sequence:
            return [state.output]
        # sink state
        if len(state.transitions.keys()) == 0:
            raise Exception('sink state')
            # return [state.output]
        return super(MooreMachine, self).compute_output_seq(state, sequence)

    def to_state_setup(self):
        state_setup_dict = {}

        # ensure prefixes are computed
        self.compute_prefixes()

        sorted_states = sorted(self.states, key=lambda x: len(x.prefix))
        for s in sorted_states:
            state_setup_dict[s.state_id] = (s.output, {k: v.state_id for k, v in s.transitions.items()})

        return state_setup_dict
