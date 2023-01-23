import itertools
import random
from abc import ABC, abstractmethod
from collections import defaultdict
from typing import Optional

import networkx
from aalpy.automata import MooreMachine, Dfa
from aalpy.utils import dfa_from_state_setup, generate_random_moore_machine, generate_random_dfa, moore_from_state_setup
from func_timeout import func_timeout, FunctionTimedOut
from six import unichr
from termcolor import colored


class Example(ABC):
    @abstractmethod
    def create_MAX_SEQ_LEN(self) -> int:
        pass

    @abstractmethod
    def create_specification_dfa(self) -> Optional[Dfa]:
        pass

    @abstractmethod
    def create_input_alphabet(self) -> list:
        pass

    @abstractmethod
    def create_output_alphabet(self) -> list:
        pass

    @abstractmethod
    def create_blackbox(self, make_in_complete=False) -> MooreMachine:
        pass

    @abstractmethod
    def create_bModelCheck(self) -> bool:
        pass

    @abstractmethod
    def create_bBBCheck(self) -> bool:
        pass

    # @abstractmethod
    # def create_all_input_permutations(self):
    #     pass
    @staticmethod
    def generate_dfa_from_regex(regex, alphabet):
        """
        Learn a regular expression.
        :param regex: regex to learn
        :param alphabet: alphabet of the regex
        :return: DFA representing the regex
        """
        from aalpy.SULs import RegexSUL
        from aalpy.oracles import StatePrefixEqOracle
        from aalpy.learning_algs import run_Lstar

        regex_sul = RegexSUL(regex)

        eq_oracle = StatePrefixEqOracle(alphabet, regex_sul, walks_per_state=2000,
                                        walk_len=15)

        learned_regex = run_Lstar(alphabet, regex_sul, eq_oracle, automaton_type='dfa')

        for s in learned_regex.states:
            if s.is_accepting:
                for a in alphabet:
                    s.transitions[a] = s
        # learned_regex.visualize()

        return learned_regex

    @staticmethod
    def validate_paths_in_blackbox(blackbox: MooreMachine, paths: set) -> set:
        validate_paths = paths.copy()
        for p in paths:
            curr_state = blackbox.initial_state
            for k in p:
                if k not in curr_state.transitions:
                    validate_paths.remove(p)
                    break
                else:
                    curr_state = curr_state.transitions[k]

        return validate_paths

    @staticmethod
    def to_digraph(states):
        states_names = [str(x.state_id) for x in states]

        G = networkx.DiGraph()

        G.add_nodes_from(states_names)

        for s in states:
            for action, target in s.transitions.items():
                G.add_edge(str(s.state_id), str(target.state_id), label=str(action))

        return G

    @staticmethod
    def to_digraph_cross_product(statesA, statesB):
        G = networkx.DiGraph()

        for x in statesA:
            for y in statesB:
                name = str(x.state_id) + str(y.state_id)
                # G.add_node(name)
                for action, target in x.transitions.items():
                    # statesB - specification is always complete so no validation for existing action in y
                    name_target = str(target.state_id) + str(y.transitions[action].state_id)
                    G.add_edge(name, str(name_target), label=str(action))
        return G

    @staticmethod
    def get_shortest_path_to_accept(dfa: Dfa):
        accepts = [x for x in dfa.states if x.is_accepting]

        if len(accepts) == 0:
            return None

        G = Example.to_digraph(dfa.states)

        path = None

        try:
            path = networkx.shortest_path(G, source=dfa.initial_state.state_id, target=accepts[0].state_id)
        except Exception as e:
            path = None
            print(e)

        accepts.pop(0)

        for s in accepts:
            try:
                path_t = networkx.shortest_path(G, source=dfa.initial_state.state_id, target=s.state_id)
                if path is None or (path_t is not None and len(path_t) < len(path)):
                    path = path_t
            except Exception as e:
                path_t = None
                print(e)

        if path is not None:
            path_s = [dfa.get_state_by_id(s) for s in path]
            actions = [[action for action, target in x.transitions.items() if target == y][0] for x, y in
                       zip(path_s[:-1], path_s[1:])]
            path = ''.join(actions)
        return path if path is not None else ''

    @staticmethod
    def get_all_path_to_accept(dfa: Dfa):
        accepts = [x for x in dfa.states if x.is_accepting]

        if len(accepts) == 0:
            return None

        G = Example.to_digraph(dfa.states)

        paths = set()
        for s in accepts:
            try:
                paths_t = networkx.all_shortest_paths(G, source=dfa.initial_state.state_id, target=s.state_id)
                if paths_t is not None:
                    for p in paths_t:
                        path_s = [dfa.get_state_by_id(s1) for s1 in p]

                        actions = [[action for action, target in x.transitions.items() if target == y][0] for x, y in
                                   zip(path_s[:-1], path_s[1:])]

                        paths.add(''.join(actions))
                # paths.update(path_t)
            except Exception as e:
                print(e)

        return list(paths)

    @staticmethod
    def get_blacklist_states_4_spec(specification_dfa: Dfa):
        blacklist_states = []
        accepting_states = []
        for s in specification_dfa.states:
            if s.is_accepting:
                accepting_states.append(s)

        for s in specification_dfa.states:
            if not s.is_accepting:
                black_state = s
                for acc in accepting_states:
                    if specification_dfa.get_shortest_path(origin_state=s, target_state=acc):
                        black_state = None
                        break
                if black_state is not None:
                    blacklist_states.append(black_state)
        return blacklist_states

    # @staticmethod
    # def get_all_paths_2_accepting_states(specification_dfa: Dfa) -> set:
    #     # accepting_states = []
    #     # all_possible_paths_specification = set()
    #     # for s in specification_dfa.states:
    #     #     if s.is_accepting:
    #     #         accepting_states.append(s)
    #     # for s in accepting_states:
    #     #     all_possible_paths_specification = all_possible_paths_specification.union(
    #     #         specification_dfa.get_all_paths(origin_state=specification_dfa.initial_state,
    #     #                                         target_state=s))
    #     # return all_possible_paths_specification
    #     g = helper.Graph(specification_dfa=specification_dfa)
    #     all_possible_paths_specification = g.getAllPaths()
    #     return all_possible_paths_specification


class Substrings(Example):

    def __init__(self):
        self.specification_dfa = None
        self.MAX_SEQ_LEN = 10
        self.blackbox = self.create_blackbox(make_in_complete=False)

    def create_MAX_SEQ_LEN(self) -> int:
        if self.blackbox is not None:
            for s in self.blackbox.states:
                if s.output == 'f':
                    p = ''.join(self.blackbox.get_shortest_path(
                        origin_state=self.blackbox.initial_state, target_state=s))
                    self.MAX_SEQ_LEN = len(p) + 2
                    return self.MAX_SEQ_LEN

    def create_specification_dfa(self) -> Optional[Dfa]:
        return None

    def create_input_alphabet(self) -> list:
        return self.blackbox.get_input_alphabet()

    def create_output_alphabet(self) -> list:
        return self.blackbox.get_output_alphabet()

    def create_blackbox(self, make_in_complete=False) -> MooreMachine:

        state_setup = {'0': ('0', {'a': '15', 'b': '11', 'c': '2'}), '1': ('5', {'a': '15', 'b': '1', 'c': '1'}),
                       '2': ('0', {'a': '15', 'b': '1', 'c': '3'}), '3': ('0', {'a': '15', 'b': '4', 'c': '15'}),
                       '4': ('4', {'a': '5', 'b': '1', 'c': '1'}), '5': ('4', {'a': '15', 'b': '6', 'c': '1'}),
                       '6': ('4', {'a': '15', 'b': '1', 'c': '7'}), '7': ('4', {'a': '8', 'b': '1', 'c': '1'}),
                       '8': ('f', {'a': '15', 'b': '10', 'c': '9'}), '9': ('4', {'a': '9', 'b': '9', 'c': '9'}),
                       '10': ('3', {'a': '15', 'b': '1', 'c': '1'}), '11': ('2', {'a': '15', 'b': '1', 'c': '12'}),
                       '12': ('2', {'a': '13', 'b': '1', 'c': '1'}), '13': ('3', {'a': '15', 'b': '14', 'c': '1'}),
                       '14': ('3', {'a': '15', 'b': '1', 'c': '10'}), '15': ('5', {'a': '15', 'b': '10', 'c': '1'})}

        self.blackbox = moore_from_state_setup(state_setup=state_setup)

        return self.blackbox

    def create_bModelCheck(self) -> bool:
        return False

    def create_bBBCheck(self) -> bool:
        return True


class ZeroOneTwo(Example):

    def __init__(self):
        self.specification_dfa = None
        self.MAX_SEQ_LEN = 10
        self.blackbox = self.create_blackbox(make_in_complete=False)

    def create_MAX_SEQ_LEN(self) -> int:
        if self.blackbox is not None:
            for s in self.blackbox.states:
                if s.output == 'f':
                    p = ''.join(self.blackbox.get_shortest_path(
                        origin_state=self.blackbox.initial_state, target_state=s))
                    self.MAX_SEQ_LEN = len(p) + 2
                    return self.MAX_SEQ_LEN

    def create_specification_dfa(self) -> Optional[Dfa]:
        return None

    def create_input_alphabet(self) -> list:
        return self.blackbox.get_input_alphabet()

    def create_output_alphabet(self) -> list:
        return self.blackbox.get_output_alphabet()

    def create_blackbox(self, make_in_complete=False) -> MooreMachine:

        state_setup = {'1': ('0', {'0': '2', '1': '6', '2': '1_', '3': '1*', '4': '1', '5': '2'}),
                       '2': ('4', {'0': '3', '1': '7', '2': '2_', '3': '2*', '4': '1', '5': '2'}),
                       '3': ('3', {'0': '4', '1': '8', '2': '3_', '3': '3*', '4': '1', '5': '2'}),
                       '4': ('2', {'0': '5', '1': '9', '2': '4_', '3': '4*', '4': '1', '5': '2'}),
                       '5': ('1', {'0': '1', '1': '10', '2': '5_', '3': '5*', '4': '1', '5': '2'}),
                       '6': ('2', {'0': '7', '1': '11', '2': '6_', '3': '6*', '4': '1', '5': '2'}),
                       '7': ('6', {'0': '8', '1': '12', '2': '7_', '3': '7*', '4': '1', '5': '2'}),
                       '8': ('5', {'0': '9', '1': '13', '2': '8_', '3': '8*', '4': '1', '5': '2'}),
                       '9': ('4', {'0': '10', '1': '14', '2': '9_', '3': '9*', '4': '1', '5': '2'}),
                       '10': ('3', {'0': '6', '1': '15', '2': '10_', '3': '10*', '4': '1', '5': '2'}),
                       '11': ('1', {'0': '12', '1': '1', '2': '11_', '3': '11*', '4': '1', '5': '2'}),
                       '12': ('5', {'0': '13', '1': '2', '2': '12_', '3': '12*', '4': '1', '5': '2'}),
                       '13': ('4', {'0': '14', '1': '3', '2': '13_', '3': '13*', '4': '1', '5': '2'}),
                       '14': ('3', {'0': '15', '1': '4', '2': '14_', '3': '14*', '4': '1', '5': '2'}),
                       '15': ('2', {'0': '11', '1': '5', '2': '15_', '3': '15*', '4': '1', '5': '2'}),
                       '1_': ('1', {'0': '2_', '1': '6_', '2': '1', '3': '1_*', '4': '1', '5': '2'}),
                       '2_': ('5', {'0': '3_', '1': '7_', '2': '2', '3': '2_*', '4': '1', '5': '2'}),
                       '3_': ('4', {'0': '4_', '1': '8_', '2': '3', '3': '3_*', '4': '1', '5': '2'}),
                       '4_': ('3', {'0': '5_', '1': '9_', '2': '4', '3': '4_*', '4': '1', '5': '2'}),
                       '5_': ('2', {'0': '1_', '1': '10_', '2': '5', '3': '5_*', '4': '1', '5': '2'}),
                       '6_': ('3', {'0': '7_', '1': '11_', '2': '6', '3': '6_*', '4': '1', '5': '2'}),
                       '7_': ('7', {'0': '8_', '1': '12_', '2': '7', '3': '7_*', '4': '1', '5': '2'}),
                       '8_': ('6', {'0': '9_', '1': '13_', '2': '8', '3': '8_*', '4': '1', '5': '2'}),
                       '9_': ('5', {'0': '10_', '1': '14_', '2': '9', '3': '9_*', '4': '1', '5': '2'}),
                       '10_': ('4', {'0': '6_', '1': '15_', '2': '10', '3': '10_*', '4': '1', '5': '2'}),
                       '11_': ('2', {'0': '12_', '1': '1_', '2': '11', '3': '11_*', '4': '1', '5': '2'}),
                       '12_': ('6', {'0': '13_', '1': '2_', '2': '12', '3': '12_*', '4': '1', '5': '2'}),
                       '13_': ('5', {'0': '14_', '1': '3_', '2': '13', '3': '13_*', '4': '1', '5': '2'}),
                       '14_': ('4', {'0': '15_', '1': '4_', '2': '14', '3': '14_*', '4': '1', '5': '2'}),
                       '15_': ('3', {'0': '11_', '1': '5_', '2': '15', '3': '15_*', '4': '1', '5': '2'}),
                       '1*': ('1', {'0': '2*', '1': '6*', '2': '1_*', '3': '1', '4': '1', '5': '2'}),
                       '2*': ('5', {'0': '3*', '1': '7*', '2': '2_*', '3': '2', '4': '1', '5': '2'}),
                       '3*': ('4', {'0': '4*', '1': '8*', '2': '3_*', '3': '3', '4': '1', '5': '2'}),
                       '4*': ('3', {'0': '5*', '1': '9*', '2': '4_*', '3': '4', '4': '1', '5': '2'}),
                       '5*': ('2', {'0': '1*', '1': '10*', '2': '5_*', '3': '5', '4': '1', '5': '2'}),
                       '6*': ('3', {'0': '7*', '1': '11*', '2': '6_*', '3': '6', '4': '1', '5': '2'}),
                       '7*': ('7', {'0': '8*', '1': '12*', '2': '7_*', '3': '7', '4': '1', '5': '2'}),
                       '8*': ('6', {'0': '9*', '1': '13*', '2': '8_*', '3': '8', '4': '1', '5': '2'}),
                       '9*': ('5', {'0': '10*', '1': '14*', '2': '9_*', '3': '9', '4': '1', '5': '2'}),
                       '10*': ('4', {'0': '6*', '1': '15*', '2': '10_*', '3': '10', '4': '1', '5': '2'}),
                       '11*': ('2', {'0': '12*', '1': '1*', '2': '11_*', '3': '11', '4': '1', '5': '2'}),
                       '12*': ('6', {'0': '13*', '1': '2*', '2': '12_*', '3': '12', '4': '1', '5': '2'}),
                       '13*': ('5', {'0': '14*', '1': '3*', '2': '13_*', '3': '13', '4': '1', '5': '2'}),
                       '14*': ('4', {'0': '15*', '1': '4*', '2': '14_*', '3': '14', '4': '1', '5': '2'}),
                       '15*': ('3', {'0': '11*', '1': '5*', '2': '15_*', '3': '15', '4': '1', '5': '2'}),
                       '1_*': ('2', {'0': '2_*', '1': '6_*', '2': '1*', '3': '1_', '4': '1', '5': '2'}),
                       '2_*': ('6', {'0': '3_*', '1': '7_*', '2': '2*', '3': '2_', '4': '1', '5': '2'}),
                       '3_*': ('5', {'0': '4_*', '1': '8_*', '2': '3*', '3': '3_', '4': '1', '5': '2'}),
                       '4_*': ('4', {'0': '5_*', '1': '9_*', '2': '4*', '3': '4_', '4': '1', '5': '2'}),
                       '5_*': ('3', {'0': '1_*', '1': '10_*', '2': '5*', '3': '5_', '4': '1', '5': '2'}),
                       '6_*': ('4', {'0': '7_*', '1': '11_*', '2': '6*', '3': '6_', '4': '1', '5': '2'}),
                       '7_*': ('8', {'0': '8_*', '1': '12_*', '2': '7*', '3': '7_', '4': '1', '5': '2'}),
                       '8_*': ('7', {'0': '9_*', '1': '13_*', '2': '8*', '3': '8_', '4': '1', '5': '2'}),
                       '9_*': ('6', {'0': '10_*', '1': '14_*', '2': '9*', '3': '9_', '4': '1', '5': '2'}),
                       '10_*': ('5', {'0': '6_*', '1': '15_*', '2': '10*', '3': '10_', '4': '1', '5': '2'}),
                       '11_*': ('3', {'0': '12_*', '1': '1_*', '2': '11*', '3': '11_', '4': '1', '5': '2'}),
                       '12_*': ('7', {'0': '13_*', '1': '2_*', '2': '12*', '3': '12_', '4': '1', '5': '2'}),
                       '13_*': ('6', {'0': '14_*', '1': '3_*', '2': '13*', '3': '13_', '4': '1', '5': '2'}),
                       '14_*': ('f', {'0': '15_*', '1': '4_*', '2': '14*', '3': '14_', '4': '1', '5': '2'}),
                       '15_*': ('4', {'0': '11_*', '1': '5_*', '2': '15*', '3': '15_', '4': '1', '5': '2'})}

        self.blackbox = moore_from_state_setup(state_setup=state_setup)

        return self.blackbox

    def create_bModelCheck(self) -> bool:
        return False

    def create_bBBCheck(self) -> bool:
        return True


class SuffixOne(Example):

    def __init__(self):
        self.specification_dfa = None
        self.MAX_SEQ_LEN = 10
        self.blackbox = self.create_blackbox(make_in_complete=False)

    def create_MAX_SEQ_LEN(self) -> int:
        if self.blackbox is not None:
            for s in self.blackbox.states:
                if s.output == 'f':
                    p = ''.join(self.blackbox.get_shortest_path(
                        origin_state=self.blackbox.initial_state, target_state=s))
                    self.MAX_SEQ_LEN = len(p) + 2
                    return self.MAX_SEQ_LEN

    def create_specification_dfa(self) -> Optional[Dfa]:
        return None

    def create_input_alphabet(self) -> list:
        return self.blackbox.get_input_alphabet()

    def create_output_alphabet(self) -> list:
        return self.blackbox.get_output_alphabet()

    def create_blackbox(self, make_in_complete=False) -> MooreMachine:

        state_setup = {'0': ('2', {'0': '1', '1': '0', '2': '10', '3': '10'}),
                       '1': ('3', {'0': '3', '1': '2', '2': '10', '3': '10'}),
                       '2': ('3', {'0': '4', '1': '0', '2': '10', '3': '10'}),
                       '3': ('4', {'0': '6', '1': '5', '2': '10', '3': '10'}),
                       '4': ('4', {'0': '7', '1': '2', '2': '10', '3': '10'}),
                       '5': ('4', {'0': '8', '1': '0', '2': '10', '3': '10'}),
                       '6': ('5', {'0': '11', '1': '9', '2': '10', '3': '10'}),
                       '7': ('5', {'0': '12', '1': '5', '2': '10', '3': '10'}),
                       '8': ('5', {'0': '13', '1': '2', '2': '10', '3': '10'}),
                       '9': ('5', {'0': '14', '1': '0', '2': '10', '3': '10'}),
                       '10': ('0', {'0': '10', '1': '21', '2': '10', '3': '10'}),
                       '11': ('f', {'0': '16', '1': '15', '2': '10', '3': '10'}),
                       '12': ('6', {'0': '17', '1': '9', '2': '10', '3': '10'}),
                       '13': ('6', {'0': '18', '1': '5', '2': '10', '3': '10'}),
                       '14': ('6', {'0': '19', '1': '2', '2': '10', '3': '10'}),
                       '15': ('6', {'0': '20', '1': '0', '2': '10', '3': '10'}),
                       '16': ('6', {'0': '0', '1': '21', '2': '10', '3': '10'}),
                       '17': ('5', {'0': '16', '1': '15', '2': '10', '3': '10'}),
                       '18': ('4', {'0': '17', '1': '9', '2': '10', '3': '10'}),
                       '19': ('3', {'0': '18', '1': '5', '2': '10', '3': '10'}),
                       '20': ('2', {'0': '19', '1': '2', '2': '10', '3': '10'}),
                       '21': ('1', {'0': '20', '1': '0', '2': '10', '3': '10'})}

        self.blackbox = moore_from_state_setup(state_setup=state_setup)

        return self.blackbox

    def create_bModelCheck(self) -> bool:
        return False

    def create_bBBCheck(self) -> bool:
        return True


class Parenthesis(Example):

    def __init__(self):
        self.specification_dfa = None
        self.MAX_SEQ_LEN = 10
        self.blackbox = self.create_blackbox(make_in_complete=False)

    def create_MAX_SEQ_LEN(self) -> int:
        if self.blackbox is not None:
            for s in self.blackbox.states:
                if s.output == 'f':
                    p = ''.join(self.blackbox.get_shortest_path(
                        origin_state=self.blackbox.initial_state, target_state=s))
                    self.MAX_SEQ_LEN = len(p) + 2
                    return self.MAX_SEQ_LEN

    def create_specification_dfa(self) -> Optional[Dfa]:
        return None

    def create_input_alphabet(self) -> list:
        return self.blackbox.get_input_alphabet()

    def create_output_alphabet(self) -> list:
        return self.blackbox.get_output_alphabet()

    def create_blackbox(self, make_in_complete=False) -> MooreMachine:

        state_setup = {'0': ('valid', {'(': '17', ')': '17', '[': '1', ']': '17'}),
                       '1': ('5', {'(': '2', ')': '17', '[': '5', ']': '0'}),
                       '5': ('5', {'(': '6', ')': '17', '[': '9', ']': '1'}),
                       '9': ('5', {'(': '10', ')': '17', '[': '13', ']': '5'}),
                       '13': ('5', {'(': '14', ')': '17', '[': '17', ']': '9'}),
                       '4': ('n', {'(': '4', ')': '4', '[': '4', ']': '4'}),
                       '2': ('close', {'(': '3', ')': '1', '[': '4', ']': '4'}),
                       '3': ('close', {'(': '4', ')': '2', '[': '4', ']': '4'}),
                       '8': ('n', {'(': '8', ')': '8', '[': '8', ']': '8'}),
                       '6': ('close', {'(': '7', ')': '5', '[': '8', ']': '8'}),
                       '7': ('close', {'(': '8', ')': '6', '[': '8', ']': '8'}),
                       '12': ('n', {'(': '12', ')': '12', '[': '12', ']': '12'}),
                       '10': ('close', {'(': '11', ')': '9', '[': '12', ']': '12'}),
                       '11': ('close', {'(': '12', ')': '10', '[': '12', ']': '12'}),
                       '16': ('n', {'(': '16', ')': '16', '[': '16', ']': '16'}),
                       '14': ('close', {'(': '15', ')': '13', '[': '16', ']': '16'}),
                       '15': ('f', {'(': '16', ')': '14', '[': '16', ']': '16'}),
                       '17': ('n', {'(': '17', ')': '17', '[': '17', ']': '17'})}

        self.blackbox = moore_from_state_setup(state_setup=state_setup)

        return self.blackbox

    def create_bModelCheck(self) -> bool:
        return False

    def create_bBBCheck(self) -> bool:
        return True


class Divisibility(Example):

    def __init__(self):
        self.specification_dfa = None
        self.MAX_SEQ_LEN = 10
        self.blackbox = self.create_blackbox(make_in_complete=False)

    def create_MAX_SEQ_LEN(self) -> int:
        if self.blackbox is not None:
            for s in self.blackbox.states:
                if s.output == 'f':
                    p = ''.join(self.blackbox.get_shortest_path(
                        origin_state=self.blackbox.initial_state, target_state=s))
                    self.MAX_SEQ_LEN = len(p) + 2
                    return self.MAX_SEQ_LEN

    def create_specification_dfa(self) -> Optional[Dfa]:
        return None

    def create_input_alphabet(self) -> list:
        return self.blackbox.get_input_alphabet()

    def create_output_alphabet(self) -> list:
        return self.blackbox.get_output_alphabet()

    def create_blackbox(self, make_in_complete=False) -> MooreMachine:

        state_setup = {'0': ('0', {'0': '1', '1': '35'}), '1': ('n', {'0': '2', '1': '5'}),
                       '2': ('n', {'0': '3', '1': '4'}), '3': ('0', {'0': '3', '1': '3'}),
                       '4': ('1', {'0': '4', '1': '4'}), '5': ('n', {'0': '6', '1': '8'}),
                       '6': ('2', {'0': '6', '1': '7'}), '7': ('0', {'0': '6', '1': '7'}),
                       '8': ('3', {'0': '8', '1': '9'}), '9': ('0', {'0': '10', '1': '8'}),
                       '10': ('0', {'0': '11', '1': '10'}), '11': ('0', {'0': '12', '1': '8'}),
                       '12': ('0', {'0': '36', '1': '10'}), '13': ('n', {'0': '14', '1': '17'}),
                       '14': ('4', {'0': '14', '1': '15'}), '15': ('0', {'0': '16', '1': '15'}),
                       '16': ('0', {'0': '14', '1': '15'}), '17': ('5', {'0': '17', '1': '18'}),
                       '18': ('0', {'0': '21', '1': '19'}), '19': ('0', {'0': '18', '1': '21'}),
                       '20': ('0', {'0': '19', '1': '20'}), '21': ('0', {'0': '20', '1': '17'}),
                       '22': ('n', {'0': '23', '1': '28'}), '23': ('6', {'0': '23', '1': '24'}),
                       '24': ('0', {'0': '26', '1': '25'}), '25': ('0', {'0': '23', '1': '24'}),
                       '26': ('0', {'0': '27', '1': '26'}), '27': ('0', {'0': '26', '1': '25'}),
                       '28': ('7', {'0': '28', '1': '29'}), '29': ('0', {'0': '30', '1': '32'}),
                       '30': ('0', {'0': '31', '1': '34'}), '31': ('0', {'0': '29', '1': '30'}),
                       '32': ('0', {'0': '33', '1': '28'}), '33': ('0', {'0': '34', '1': '33'}),
                       '34': ('0', {'0': '32', '1': '31'}), '35': ('n', {'0': '13', '1': '22'}),
                       '36': ('0', {'0': '37', '1': '8'}), '37': ('0', {'0': '38', '1': '8'}),
                       '38': ('0', {'0': '39', '1': '8'}), '39': ('0', {'0': '40', '1': '8'}),
                       '40': ('f', {'0': '40', '1': '40'})}

        self.blackbox = moore_from_state_setup(state_setup=state_setup)

        return self.blackbox

    def create_bModelCheck(self) -> bool:
        return False

    def create_bBBCheck(self) -> bool:
        return True


class EqualStub(Example):

    def __init__(self):
        self.specification_dfa = None
        self.MAX_SEQ_LEN = 10
        self.blackbox = self.create_blackbox(make_in_complete=False)

    def create_MAX_SEQ_LEN(self) -> int:
        if self.blackbox is not None:
            for s in self.blackbox.states:
                if s.output == 'f':
                    p = ''.join(self.blackbox.get_shortest_path(
                        origin_state=self.blackbox.initial_state, target_state=s))
                    self.MAX_SEQ_LEN = len(p) + 2
                    return self.MAX_SEQ_LEN

    def create_specification_dfa(self) -> Optional[Dfa]:
        return None

    def create_input_alphabet(self) -> list:
        return self.blackbox.get_input_alphabet()

    def create_output_alphabet(self) -> list:
        return self.blackbox.get_output_alphabet()

    def create_blackbox(self, make_in_complete=False) -> MooreMachine:

        state_setup = {
            '0': ('6', {'0': '24', '1': '23', '2': '22', '3': '21'}),
            '1': ('2', {'0': '4', '1': '4', '2': '4', '3': '2'}),
            '2': ('1', {'0': '4', '1': '4', '2': '4', '3': '3'}),
            '3': ('0', {'0': '4', '1': '4', '2': '4', '3': '3'}),
            '4': ('3', {'0': '4', '1': '4', '2': '4', '3': '1'}),
            '5': ('4', {'0': '25', '1': '25', '2': '25', '3': '4'}),
            '6': ('3', {'0': '6', '1': '6', '2': '7', '3': '6'}),
            '7': ('2', {'0': '6', '1': '6', '2': '8', '3': '6'}),
            '8': ('1', {'0': '6', '1': '6', '2': '9', '3': '6'}),
            '9': ('0', {'0': '6', '1': '6', '2': '9', '3': '6'}),
            '10': ('4', {'0': '25', '1': '25', '2': '6', '3': '25'}),
            '11': ('3', {'0': '11', '1': '12', '2': '11', '3': '11'}),
            '12': ('2', {'0': '11', '1': '13', '2': '11', '3': '11'}),
            '13': ('1', {'0': '11', '1': '14', '2': '11', '3': '11'}),
            '14': ('0', {'0': '11', '1': '14', '2': '11', '3': '11'}),
            '15': ('4', {'0': '25', '1': '11', '2': '25', '3': '25'}),
            '16': ('3', {'0': '17', '1': '16', '2': '16', '3': '16'}),
            '17': ('2', {'0': '18', '1': '16', '2': '16', '3': '16'}),
            '18': ('1', {'0': '19', '1': '16', '2': '16', '3': '16'}),
            '19': ('f', {'0': '19', '1': '16', '2': '16', '3': '16'}),
            '20': ('4', {'0': '16', '1': '25', '2': '25', '3': '25'}),
            '21': ('5', {'0': '25', '1': '25', '2': '25', '3': '5'}),
            '22': ('5', {'0': '25', '1': '25', '2': '10', '3': '25'}),
            '23': ('5', {'0': '25', '1': '15', '2': '25', '3': '25'}),
            '24': ('5', {'0': '20', '1': '25', '2': '25', '3': '25'}),
            '25': ('999', {'0': '25', '1': '25', '2': '25', '3': '25'}),
        }

        self.blackbox = moore_from_state_setup(state_setup=state_setup)

        return self.blackbox

    def create_bModelCheck(self) -> bool:
        return False

    def create_bBBCheck(self) -> bool:
        return True


class ModelCheckingSpecRegex(Example):
    def __init__(self):
        self.MAX_SEQ_LEN = 10
        self.specification_dfa = None
        self.num_of_states_specification = 10
        self.num_of_states_bb = 10
        self.input_alphabet = self.create_input_alphabet()
        self.output_alphabet = self.create_output_alphabet()

    def create_MAX_SEQ_LEN(self) -> int:
        if self.specification_dfa is not None:
            for s in self.specification_dfa.states:
                if s.is_accepting:
                    p = ''.join(self.specification_dfa.get_shortest_path(
                        origin_state=self.specification_dfa.initial_state, target_state=s))
                    return len(p) + 2

    def create_specification_dfa(self) -> Optional[Dfa]:
        # specification_dfa = dfa_from_state_setup(self.dfa_state_setup)
        if self.specification_dfa is None:
            # self.specification_dfa = self.generate_dfa_from_regex('aaaaa*+(bb+cc)dab+ddddd', alphabet=self.input_alphabet)
            self.specification_dfa = self.generate_dfa_from_regex('aaaaa*|(bb|cc)dab|ddddd',
                                                                  alphabet=self.input_alphabet)

        if self.input_alphabet is None:
            self.input_alphabet = self.specification_dfa.get_input_alphabet()
        return self.specification_dfa

    def create_input_alphabet(self) -> list:
        self.input_alphabet = ['a', 'b', 'c', 'd']
        return self.input_alphabet

    def create_output_alphabet(self) -> list:
        self.output_alphabet = [str(unichr(97 + x)) for x in range(0, round(self.num_of_states_bb * 1 / 4))]
        return self.output_alphabet

    def create_blackbox(self, make_in_complete=False) -> MooreMachine:
        if self.input_alphabet is None:
            self.create_input_alphabet()

        if self.output_alphabet is None:
            self.create_output_alphabet()

        # randomly generate blackbox automaton
        blackbox = generate_random_moore_machine(num_states=self.num_of_states_bb, input_alphabet=self.input_alphabet,
                                                 output_alphabet=self.output_alphabet)
        return blackbox

    def create_bModelCheck(self) -> bool:
        return True

    def create_bBBCheck(self) -> bool:
        return False


class ModelCheckingRandom(Example):
    # dfa_state_setup = {
    #     'q0': (False, {'a': 'q1', 'b': 'q0'}),
    #     'q1': (False, {'a': 'q2', 'b': 'q0'}),
    #     'q2': (False, {'a': 'q3', 'b': 'q0'}),
    #     'q3': (True, {'a': 'q3', 'b': 'q3'})
    # }

    # dfa_state_setup = {}
    # lengh = 8
    # for i in range(lengh):
    #     curr_q = 'q' + str(i)
    #     next_q = f'q{i + 1}'
    #     start_q = 'q0'
    #
    #     if i == lengh - 1:
    #         dfa_state_setup[curr_q] = (True, {'a': curr_q, 'b': curr_q})
    #     else:
    #         dfa_state_setup[curr_q] = (False, {'a': next_q, 'b': start_q})

    def __init__(self):
        self.MAX_SEQ_LEN = 10
        self.specification_dfa = None
        self.num_of_states_specification = 200
        self.num_of_states_bb = 10
        self.input_alphabet = self.create_input_alphabet()
        self.output_alphabet = self.create_output_alphabet()

    def create_MAX_SEQ_LEN(self) -> int:
        return self.MAX_SEQ_LEN

    def create_specification_dfa(self, force=False) -> Optional[Dfa]:
        # specification_dfa = dfa_from_state_setup(self.dfa_state_setup)
        if self.specification_dfa is None or force:

            generation_failed = True
            while generation_failed:
                try:
                    while True:
                        self.specification_dfa = generate_random_dfa(num_states=self.num_of_states_specification,
                                                                     alphabet=self.input_alphabet,
                                                                     ensure_minimality=True,
                                                                     safety=True)
                        max_len = 0
                        max_len_s = self.specification_dfa.initial_state
                        for s in self.specification_dfa.states:
                            s.is_accepting = False
                            p = self.specification_dfa.get_shortest_path(
                                origin_state=self.specification_dfa.initial_state, target_state=s)
                            if p and len(''.join(p)) > max_len:
                                max_len = len(''.join(p))
                                max_len_s = s

                        max_len_s.is_accepting = True
                        self.MAX_SEQ_LEN = max_len * 2
                        if not self.specification_dfa.initial_state.is_accepting:
                            generation_failed = False
                            break

                except FunctionTimedOut:
                    print(colored('error generating', 'red'))
                    generation_failed = True
                except Exception as e:
                    print(e)
                    print(colored('error generating', 'red'))
                    generation_failed = True

        if self.input_alphabet is None:
            self.input_alphabet = self.specification_dfa.get_input_alphabet()
        return self.specification_dfa

    def create_input_alphabet(self) -> list:
        self.input_alphabet = [str(x) for x in range(0, round(self.num_of_states_bb * 1 / 10 + 1))]
        return self.input_alphabet

    def create_output_alphabet(self) -> list:
        self.output_alphabet = [str(unichr(97 + x)) for x in range(0, round(self.num_of_states_bb * 3 / 4))]
        return self.output_alphabet

    def create_blackbox(self, make_in_complete=False) -> MooreMachine:
        if self.input_alphabet is None:
            self.create_input_alphabet()

        if self.output_alphabet is None:
            self.create_output_alphabet()

        # randomly generate blackbox automaton
        blackbox = generate_random_moore_machine(num_states=self.num_of_states_bb, input_alphabet=self.input_alphabet,
                                                 output_alphabet=self.output_alphabet)

        while make_in_complete:
            make_in_complete = False
            # paths = list(self.specification_dfa.get_all_paths(origin_state=self.specification_dfa.initial_state,
            #                                                   target_state=accepting_state))

            paths = Example.get_all_path_to_accept(dfa=self.specification_dfa)
            if len(paths) == 1:
                make_in_complete = True
                self.specification_dfa = self.create_specification_dfa(force=True)
                continue

            paths_remove = random.choices(paths, k=1)
            for p in paths_remove:
                curr_state = blackbox.initial_state
                for k in p[:-1]:
                    if k in curr_state.transitions:
                        curr_state = curr_state.transitions[k]
                    else:
                        break
                c = curr_state.transitions.copy()
                for a in c:
                    del curr_state.transitions[a]
                curr_state.output = 'sink'

            paths_left = [p for p in paths if p not in paths_remove]
            counter_not_valid = 0
            for p in paths_left:
                try:
                    blackbox.execute_sequence(origin_state=blackbox.initial_state, seq=p)
                except:
                    counter_not_valid += 1

            if counter_not_valid == len(paths):
                make_in_complete = True

        return blackbox

    def create_bModelCheck(self) -> bool:
        return True

    def create_bBBCheck(self) -> bool:
        return False


class ModelCheckingRandomBigSpecBBDivisibility(Example):
    def __init__(self, bb_size=None, spec_size=None):
        self.MAX_SEQ_LEN = 10
        self.specification_dfa = None
        self.num_of_states_specification = 100 if spec_size is None else spec_size
        self.num_of_states_bb = 20 if bb_size is None else bb_size
        self.input_alphabet = self.create_input_alphabet()
        self.output_alphabet = self.create_output_alphabet()

    def create_MAX_SEQ_LEN(self) -> int:
        return self.MAX_SEQ_LEN

    def create_specification_dfa(self, force=False) -> Optional[Dfa]:
        # specification_dfa = dfa_from_state_setup(self.dfa_state_setup)
        if self.specification_dfa is None or force:

            generation_failed = True
            while generation_failed:
                try:
                    while True:
                        self.specification_dfa = generate_random_dfa(num_states=self.num_of_states_specification,
                                                                     alphabet=self.input_alphabet,
                                                                     ensure_minimality=True,
                                                                     safety=False)
                        max_len = 0
                        max_len_s = self.specification_dfa.initial_state
                        for s in self.specification_dfa.states:
                            s.is_accepting = False
                            p = self.specification_dfa.get_shortest_path(
                                origin_state=self.specification_dfa.initial_state, target_state=s)
                            if p and len(''.join(p)) > max_len:
                                max_len = len(''.join(p))
                                max_len_s = s

                        setup = {}
                        # print(max_len_s.state_id)
                        MAX_REPEAT = 4
                        for i in range(1, MAX_REPEAT + 1):
                            for s in self.specification_dfa.states:
                                setup_state = {}
                                name = s.state_id + '_' + str(i)

                                if s != max_len_s:
                                    for a, t in s.transitions.items():
                                        setup_state[a] = t.state_id + '_' + str(i)
                                    setup[name] = (False, setup_state)
                                else:

                                    if i < MAX_REPEAT:
                                        for a in s.transitions.keys():
                                            setup_state[a] = self.specification_dfa.initial_state.state_id + '_' + str(
                                                i + 1)
                                        setup[name] = (False, setup_state)
                                    else:
                                        for a in s.transitions.keys():
                                            setup_state[a] = name
                                        setup[name] = (True, setup_state)

                        self.specification_dfa = dfa_from_state_setup(state_setup=setup)

                        # force safety
                        # max_len_s.is_accepting = True
                        # for a, in max_len_s.transitions.keys():
                        #     max_len_s.transitions[a] = max_len_s
                        #     print(max_len)

                        self.MAX_SEQ_LEN = max_len * MAX_REPEAT * 2
                        if not self.specification_dfa.initial_state.is_accepting:
                            generation_failed = False
                            break

                except FunctionTimedOut:
                    print(colored('error generating', 'red'))
                    generation_failed = True
                except Exception as e:
                    print(e)
                    print(colored('error generating', 'red'))
                    generation_failed = True

        if self.input_alphabet is None:
            self.input_alphabet = self.specification_dfa.get_input_alphabet()
        return self.specification_dfa

    def create_input_alphabet(self) -> list:
        self.input_alphabet = ['0', '1']
        return self.input_alphabet

    def create_output_alphabet(self) -> list:
        self.output_alphabet = [str(unichr(97 + x)) for x in range(0, round(40 * 3 / 4))]
        return self.output_alphabet

    def create_blackbox(self, make_in_complete=False) -> MooreMachine:
        if self.input_alphabet is None:
            self.create_input_alphabet()

        if self.output_alphabet is None:
            self.create_output_alphabet()

        # randomly generate blackbox automaton

        state_setup = {'0': ('0', {'0': '1', '1': '35'}), '1': ('n', {'0': '2', '1': '5'}),
                       '2': ('n', {'0': '3', '1': '4'}), '3': ('0', {'0': '3', '1': '3'}),
                       '4': ('1', {'0': '4', '1': '4'}), '5': ('n', {'0': '6', '1': '8'}),
                       '6': ('2', {'0': '6', '1': '7'}), '7': ('0', {'0': '6', '1': '7'}),
                       '8': ('3', {'0': '8', '1': '9'}), '9': ('0', {'0': '10', '1': '8'}),
                       '10': ('0', {'0': '11', '1': '10'}), '11': ('0', {'0': '12', '1': '8'}),
                       '12': ('0', {'0': '36', '1': '10'}), '13': ('n', {'0': '14', '1': '17'}),
                       '14': ('4', {'0': '14', '1': '15'}), '15': ('0', {'0': '16', '1': '15'}),
                       '16': ('0', {'0': '14', '1': '15'}), '17': ('5', {'0': '17', '1': '18'}),
                       '18': ('0', {'0': '21', '1': '19'}), '19': ('0', {'0': '18', '1': '21'}),
                       '20': ('0', {'0': '19', '1': '20'}), '21': ('0', {'0': '20', '1': '17'}),
                       '22': ('n', {'0': '23', '1': '28'}), '23': ('6', {'0': '23', '1': '24'}),
                       '24': ('0', {'0': '26', '1': '25'}), '25': ('0', {'0': '23', '1': '24'}),
                       '26': ('0', {'0': '27', '1': '26'}), '27': ('0', {'0': '26', '1': '25'}),
                       '28': ('7', {'0': '28', '1': '29'}), '29': ('0', {'0': '30', '1': '32'}),
                       '30': ('0', {'0': '31', '1': '34'}), '31': ('0', {'0': '29', '1': '30'}),
                       '32': ('0', {'0': '33', '1': '28'}), '33': ('0', {'0': '34', '1': '33'}),
                       '34': ('0', {'0': '32', '1': '31'}), '35': ('n', {'0': '13', '1': '22'}),
                       '36': ('0', {'0': '37', '1': '8'}), '37': ('0', {'0': '38', '1': '8'}),
                       '38': ('0', {'0': '39', '1': '8'}), '39': ('0', {'0': '40', '1': '8'}),
                       '40': ('f', {'0': '40', '1': '40'})}

        blackbox = moore_from_state_setup(state_setup=state_setup)

        while make_in_complete:
            make_in_complete = False
            # paths = list(self.specification_dfa.get_all_paths(origin_state=self.specification_dfa.initial_state,
            #                                                   target_state=accepting_state))

            paths = Example.get_all_path_to_accept(dfa=self.specification_dfa)
            if len(paths) == 1:
                make_in_complete = True
                self.specification_dfa = self.create_specification_dfa(force=True)
                continue

            paths_remove = random.choices(paths, k=1)
            for p in paths_remove:
                curr_state = blackbox.initial_state
                for k in p[:-1]:
                    if k in curr_state.transitions:
                        curr_state = curr_state.transitions[k]
                    else:
                        break
                c = curr_state.transitions.copy()
                for a in c:
                    del curr_state.transitions[a]
                curr_state.output = 'sink'

            paths_left = [p for p in paths if p not in paths_remove]
            counter_not_valid = 0
            for p in paths_left:
                try:
                    blackbox.execute_sequence(origin_state=blackbox.initial_state, seq=p)
                except:
                    counter_not_valid += 1

            if counter_not_valid == len(paths):
                make_in_complete = True

        self.num_of_states_bb = len(blackbox.states)
        return blackbox

    def create_bModelCheck(self) -> bool:
        return True

    def create_bBBCheck(self) -> bool:
        return False


class ModelCheckingRandomEqualSpecBBParenthesis(Example):
    def __init__(self):
        self.MAX_SEQ_LEN = 10
        self.specification_dfa = None
        self.num_of_states_specification = 10
        self.num_of_states_bb = 10
        self.input_alphabet = self.create_input_alphabet()
        self.output_alphabet = self.create_output_alphabet()

    def create_MAX_SEQ_LEN(self) -> int:
        return self.MAX_SEQ_LEN

    def create_specification_dfa(self, force=False) -> Optional[Dfa]:
        # specification_dfa = dfa_from_state_setup(self.dfa_state_setup)
        if self.specification_dfa is None or force:

            generation_failed = True
            while generation_failed:
                try:
                    while True:
                        self.specification_dfa = generate_random_dfa(num_states=self.num_of_states_specification,
                                                                     alphabet=self.input_alphabet,
                                                                     ensure_minimality=True,
                                                                     safety=False)
                        max_len = 0
                        max_len_s = self.specification_dfa.initial_state
                        for s in self.specification_dfa.states:
                            s.is_accepting = False
                            p = self.specification_dfa.get_shortest_path(
                                origin_state=self.specification_dfa.initial_state, target_state=s)
                            if p and len(''.join(p)) > max_len:
                                max_len = len(''.join(p))
                                max_len_s = s

                        setup = {}
                        # print(max_len)
                        MAX_REPEAT = 10
                        for i in range(1, MAX_REPEAT + 1):
                            for s in self.specification_dfa.states:
                                setup_state = {}
                                name = s.state_id + '_' + str(i)

                                if s != max_len_s:
                                    for a, t in s.transitions.items():
                                        setup_state[a] = t.state_id + '_' + str(i)
                                    setup[name] = (False, setup_state)
                                else:

                                    if i < MAX_REPEAT:
                                        for a in s.transitions.keys():
                                            setup_state[a] = self.specification_dfa.initial_state.state_id + '_' + str(
                                                i + 1)
                                        setup[name] = (False, setup_state)
                                    else:
                                        for a in s.transitions.keys():
                                            setup_state[a] = name
                                        setup[name] = (True, setup_state)

                        self.specification_dfa = dfa_from_state_setup(state_setup=setup)

                        # force safety
                        # max_len_s.is_accepting = True
                        # for a, in max_len_s.transitions.keys():
                        #     max_len_s.transitions[a] = max_len_s
                        #     print(max_len)

                        self.MAX_SEQ_LEN = max_len * MAX_REPEAT * 2
                        if not self.specification_dfa.initial_state.is_accepting:
                            generation_failed = False
                            break

                except FunctionTimedOut:
                    print(colored('error generating', 'red'))
                    generation_failed = True
                except Exception as e:
                    print(e)
                    print(colored('error generating', 'red'))
                    generation_failed = True

        if self.input_alphabet is None:
            self.input_alphabet = self.specification_dfa.get_input_alphabet()
        return self.specification_dfa

    def create_input_alphabet(self) -> list:
        self.input_alphabet = ['(', ')', '[', ']']
        return self.input_alphabet

    def create_output_alphabet(self) -> list:
        self.output_alphabet = [str(unichr(97 + x)) for x in range(0, round(40 * 3 / 4))]
        return self.output_alphabet

    def create_blackbox(self, make_in_complete=False) -> MooreMachine:
        if self.input_alphabet is None:
            self.create_input_alphabet()

        if self.output_alphabet is None:
            self.create_output_alphabet()

        # build blackbox
        state_setup = {'0': ('valid', {'(': '241', ')': '241', '[': '1', ']': '241'}),
                       '1': ('5', {'(': '2', ')': '241', '[': '13', ']': '0'}),
                       '13': ('5', {'(': '14', ')': '241', '[': '25', ']': '1'}),
                       '25': ('5', {'(': '26', ')': '241', '[': '37', ']': '13'}),
                       '37': ('5', {'(': '38', ')': '241', '[': '49', ']': '25'}),
                       '49': ('5', {'(': '50', ')': '241', '[': '61', ']': '37'}),
                       '61': ('5', {'(': '62', ')': '241', '[': '73', ']': '49'}),
                       '73': ('5', {'(': '74', ')': '241', '[': '85', ']': '61'}),
                       '85': ('5', {'(': '86', ')': '241', '[': '97', ']': '73'}),
                       '97': ('5', {'(': '98', ')': '241', '[': '109', ']': '85'}),
                       '109': ('5', {'(': '110', ')': '241', '[': '121', ']': '97'}),
                       '121': ('5', {'(': '122', ')': '241', '[': '133', ']': '109'}),
                       '133': ('5', {'(': '134', ')': '241', '[': '145', ']': '121'}),
                       '145': ('5', {'(': '146', ')': '241', '[': '157', ']': '133'}),
                       '157': ('5', {'(': '158', ')': '241', '[': '169', ']': '145'}),
                       '169': ('5', {'(': '170', ')': '241', '[': '181', ']': '157'}),
                       '181': ('5', {'(': '182', ')': '241', '[': '193', ']': '169'}),
                       '193': ('5', {'(': '194', ')': '241', '[': '205', ']': '181'}),
                       '205': ('5', {'(': '206', ')': '241', '[': '217', ']': '193'}),
                       '217': ('5', {'(': '218', ')': '241', '[': '229', ']': '205'}),
                       '229': ('5', {'(': '230', ')': '241', '[': '241', ']': '217'}),
                       '12': ('n', {'(': '12', ')': '12', '[': '12', ']': '12'}),
                       '2': ('close', {'(': '3', ')': '1', '[': '12', ']': '12'}),
                       '3': ('close', {'(': '4', ')': '2', '[': '12', ']': '12'}),
                       '4': ('close', {'(': '5', ')': '3', '[': '12', ']': '12'}),
                       '5': ('close', {'(': '6', ')': '4', '[': '12', ']': '12'}),
                       '6': ('close', {'(': '7', ')': '5', '[': '12', ']': '12'}),
                       '7': ('close', {'(': '8', ')': '6', '[': '12', ']': '12'}),
                       '8': ('close', {'(': '9', ')': '7', '[': '12', ']': '12'}),
                       '9': ('close', {'(': '10', ')': '8', '[': '12', ']': '12'}),
                       '10': ('close', {'(': '11', ')': '9', '[': '12', ']': '12'}),
                       '11': ('close', {'(': '12', ')': '10', '[': '12', ']': '12'}),
                       '24': ('n', {'(': '24', ')': '24', '[': '24', ']': '24'}),
                       '14': ('close', {'(': '15', ')': '13', '[': '24', ']': '24'}),
                       '15': ('close', {'(': '16', ')': '14', '[': '24', ']': '24'}),
                       '16': ('close', {'(': '17', ')': '15', '[': '24', ']': '24'}),
                       '17': ('close', {'(': '18', ')': '16', '[': '24', ']': '24'}),
                       '18': ('close', {'(': '19', ')': '17', '[': '24', ']': '24'}),
                       '19': ('close', {'(': '20', ')': '18', '[': '24', ']': '24'}),
                       '20': ('close', {'(': '21', ')': '19', '[': '24', ']': '24'}),
                       '21': ('close', {'(': '22', ')': '20', '[': '24', ']': '24'}),
                       '22': ('close', {'(': '23', ')': '21', '[': '24', ']': '24'}),
                       '23': ('close', {'(': '24', ')': '22', '[': '24', ']': '24'}),
                       '36': ('n', {'(': '36', ')': '36', '[': '36', ']': '36'}),
                       '26': ('close', {'(': '27', ')': '25', '[': '36', ']': '36'}),
                       '27': ('close', {'(': '28', ')': '26', '[': '36', ']': '36'}),
                       '28': ('close', {'(': '29', ')': '27', '[': '36', ']': '36'}),
                       '29': ('close', {'(': '30', ')': '28', '[': '36', ']': '36'}),
                       '30': ('close', {'(': '31', ')': '29', '[': '36', ']': '36'}),
                       '31': ('close', {'(': '32', ')': '30', '[': '36', ']': '36'}),
                       '32': ('close', {'(': '33', ')': '31', '[': '36', ']': '36'}),
                       '33': ('close', {'(': '34', ')': '32', '[': '36', ']': '36'}),
                       '34': ('close', {'(': '35', ')': '33', '[': '36', ']': '36'}),
                       '35': ('close', {'(': '36', ')': '34', '[': '36', ']': '36'}),
                       '48': ('n', {'(': '48', ')': '48', '[': '48', ']': '48'}),
                       '38': ('close', {'(': '39', ')': '37', '[': '48', ']': '48'}),
                       '39': ('close', {'(': '40', ')': '38', '[': '48', ']': '48'}),
                       '40': ('close', {'(': '41', ')': '39', '[': '48', ']': '48'}),
                       '41': ('close', {'(': '42', ')': '40', '[': '48', ']': '48'}),
                       '42': ('close', {'(': '43', ')': '41', '[': '48', ']': '48'}),
                       '43': ('close', {'(': '44', ')': '42', '[': '48', ']': '48'}),
                       '44': ('close', {'(': '45', ')': '43', '[': '48', ']': '48'}),
                       '45': ('close', {'(': '46', ')': '44', '[': '48', ']': '48'}),
                       '46': ('close', {'(': '47', ')': '45', '[': '48', ']': '48'}),
                       '47': ('close', {'(': '48', ')': '46', '[': '48', ']': '48'}),
                       '60': ('n', {'(': '60', ')': '60', '[': '60', ']': '60'}),
                       '50': ('close', {'(': '51', ')': '49', '[': '60', ']': '60'}),
                       '51': ('close', {'(': '52', ')': '50', '[': '60', ']': '60'}),
                       '52': ('close', {'(': '53', ')': '51', '[': '60', ']': '60'}),
                       '53': ('close', {'(': '54', ')': '52', '[': '60', ']': '60'}),
                       '54': ('close', {'(': '55', ')': '53', '[': '60', ']': '60'}),
                       '55': ('close', {'(': '56', ')': '54', '[': '60', ']': '60'}),
                       '56': ('close', {'(': '57', ')': '55', '[': '60', ']': '60'}),
                       '57': ('close', {'(': '58', ')': '56', '[': '60', ']': '60'}),
                       '58': ('close', {'(': '59', ')': '57', '[': '60', ']': '60'}),
                       '59': ('close', {'(': '60', ')': '58', '[': '60', ']': '60'}),
                       '72': ('n', {'(': '72', ')': '72', '[': '72', ']': '72'}),
                       '62': ('close', {'(': '63', ')': '61', '[': '72', ']': '72'}),
                       '63': ('close', {'(': '64', ')': '62', '[': '72', ']': '72'}),
                       '64': ('close', {'(': '65', ')': '63', '[': '72', ']': '72'}),
                       '65': ('close', {'(': '66', ')': '64', '[': '72', ']': '72'}),
                       '66': ('close', {'(': '67', ')': '65', '[': '72', ']': '72'}),
                       '67': ('close', {'(': '68', ')': '66', '[': '72', ']': '72'}),
                       '68': ('close', {'(': '69', ')': '67', '[': '72', ']': '72'}),
                       '69': ('close', {'(': '70', ')': '68', '[': '72', ']': '72'}),
                       '70': ('close', {'(': '71', ')': '69', '[': '72', ']': '72'}),
                       '71': ('close', {'(': '72', ')': '70', '[': '72', ']': '72'}),
                       '84': ('n', {'(': '84', ')': '84', '[': '84', ']': '84'}),
                       '74': ('close', {'(': '75', ')': '73', '[': '84', ']': '84'}),
                       '75': ('close', {'(': '76', ')': '74', '[': '84', ']': '84'}),
                       '76': ('close', {'(': '77', ')': '75', '[': '84', ']': '84'}),
                       '77': ('close', {'(': '78', ')': '76', '[': '84', ']': '84'}),
                       '78': ('close', {'(': '79', ')': '77', '[': '84', ']': '84'}),
                       '79': ('close', {'(': '80', ')': '78', '[': '84', ']': '84'}),
                       '80': ('close', {'(': '81', ')': '79', '[': '84', ']': '84'}),
                       '81': ('close', {'(': '82', ')': '80', '[': '84', ']': '84'}),
                       '82': ('close', {'(': '83', ')': '81', '[': '84', ']': '84'}),
                       '83': ('close', {'(': '84', ')': '82', '[': '84', ']': '84'}),
                       '96': ('n', {'(': '96', ')': '96', '[': '96', ']': '96'}),
                       '86': ('close', {'(': '87', ')': '85', '[': '96', ']': '96'}),
                       '87': ('close', {'(': '88', ')': '86', '[': '96', ']': '96'}),
                       '88': ('close', {'(': '89', ')': '87', '[': '96', ']': '96'}),
                       '89': ('close', {'(': '90', ')': '88', '[': '96', ']': '96'}),
                       '90': ('close', {'(': '91', ')': '89', '[': '96', ']': '96'}),
                       '91': ('close', {'(': '92', ')': '90', '[': '96', ']': '96'}),
                       '92': ('close', {'(': '93', ')': '91', '[': '96', ']': '96'}),
                       '93': ('close', {'(': '94', ')': '92', '[': '96', ']': '96'}),
                       '94': ('close', {'(': '95', ')': '93', '[': '96', ']': '96'}),
                       '95': ('close', {'(': '96', ')': '94', '[': '96', ']': '96'}),
                       '108': ('n', {'(': '108', ')': '108', '[': '108', ']': '108'}),
                       '98': ('close', {'(': '99', ')': '97', '[': '108', ']': '108'}),
                       '99': ('close', {'(': '100', ')': '98', '[': '108', ']': '108'}),
                       '100': ('close', {'(': '101', ')': '99', '[': '108', ']': '108'}),
                       '101': ('close', {'(': '102', ')': '100', '[': '108', ']': '108'}),
                       '102': ('close', {'(': '103', ')': '101', '[': '108', ']': '108'}),
                       '103': ('close', {'(': '104', ')': '102', '[': '108', ']': '108'}),
                       '104': ('close', {'(': '105', ')': '103', '[': '108', ']': '108'}),
                       '105': ('close', {'(': '106', ')': '104', '[': '108', ']': '108'}),
                       '106': ('close', {'(': '107', ')': '105', '[': '108', ']': '108'}),
                       '107': ('close', {'(': '108', ')': '106', '[': '108', ']': '108'}),
                       '120': ('n', {'(': '120', ')': '120', '[': '120', ']': '120'}),
                       '110': ('close', {'(': '111', ')': '109', '[': '120', ']': '120'}),
                       '111': ('close', {'(': '112', ')': '110', '[': '120', ']': '120'}),
                       '112': ('close', {'(': '113', ')': '111', '[': '120', ']': '120'}),
                       '113': ('close', {'(': '114', ')': '112', '[': '120', ']': '120'}),
                       '114': ('close', {'(': '115', ')': '113', '[': '120', ']': '120'}),
                       '115': ('close', {'(': '116', ')': '114', '[': '120', ']': '120'}),
                       '116': ('close', {'(': '117', ')': '115', '[': '120', ']': '120'}),
                       '117': ('close', {'(': '118', ')': '116', '[': '120', ']': '120'}),
                       '118': ('close', {'(': '119', ')': '117', '[': '120', ']': '120'}),
                       '119': ('close', {'(': '120', ')': '118', '[': '120', ']': '120'}),
                       '132': ('n', {'(': '132', ')': '132', '[': '132', ']': '132'}),
                       '122': ('close', {'(': '123', ')': '121', '[': '132', ']': '132'}),
                       '123': ('close', {'(': '124', ')': '122', '[': '132', ']': '132'}),
                       '124': ('close', {'(': '125', ')': '123', '[': '132', ']': '132'}),
                       '125': ('close', {'(': '126', ')': '124', '[': '132', ']': '132'}),
                       '126': ('close', {'(': '127', ')': '125', '[': '132', ']': '132'}),
                       '127': ('close', {'(': '128', ')': '126', '[': '132', ']': '132'}),
                       '128': ('close', {'(': '129', ')': '127', '[': '132', ']': '132'}),
                       '129': ('close', {'(': '130', ')': '128', '[': '132', ']': '132'}),
                       '130': ('close', {'(': '131', ')': '129', '[': '132', ']': '132'}),
                       '131': ('close', {'(': '132', ')': '130', '[': '132', ']': '132'}),
                       '144': ('n', {'(': '144', ')': '144', '[': '144', ']': '144'}),
                       '134': ('close', {'(': '135', ')': '133', '[': '144', ']': '144'}),
                       '135': ('close', {'(': '136', ')': '134', '[': '144', ']': '144'}),
                       '136': ('close', {'(': '137', ')': '135', '[': '144', ']': '144'}),
                       '137': ('close', {'(': '138', ')': '136', '[': '144', ']': '144'}),
                       '138': ('close', {'(': '139', ')': '137', '[': '144', ']': '144'}),
                       '139': ('close', {'(': '140', ')': '138', '[': '144', ']': '144'}),
                       '140': ('close', {'(': '141', ')': '139', '[': '144', ']': '144'}),
                       '141': ('close', {'(': '142', ')': '140', '[': '144', ']': '144'}),
                       '142': ('close', {'(': '143', ')': '141', '[': '144', ']': '144'}),
                       '143': ('close', {'(': '144', ')': '142', '[': '144', ']': '144'}),
                       '156': ('n', {'(': '156', ')': '156', '[': '156', ']': '156'}),
                       '146': ('close', {'(': '147', ')': '145', '[': '156', ']': '156'}),
                       '147': ('close', {'(': '148', ')': '146', '[': '156', ']': '156'}),
                       '148': ('close', {'(': '149', ')': '147', '[': '156', ']': '156'}),
                       '149': ('close', {'(': '150', ')': '148', '[': '156', ']': '156'}),
                       '150': ('close', {'(': '151', ')': '149', '[': '156', ']': '156'}),
                       '151': ('close', {'(': '152', ')': '150', '[': '156', ']': '156'}),
                       '152': ('close', {'(': '153', ')': '151', '[': '156', ']': '156'}),
                       '153': ('close', {'(': '154', ')': '152', '[': '156', ']': '156'}),
                       '154': ('close', {'(': '155', ')': '153', '[': '156', ']': '156'}),
                       '155': ('close', {'(': '156', ')': '154', '[': '156', ']': '156'}),
                       '168': ('n', {'(': '168', ')': '168', '[': '168', ']': '168'}),
                       '158': ('close', {'(': '159', ')': '157', '[': '168', ']': '168'}),
                       '159': ('close', {'(': '160', ')': '158', '[': '168', ']': '168'}),
                       '160': ('close', {'(': '161', ')': '159', '[': '168', ']': '168'}),
                       '161': ('close', {'(': '162', ')': '160', '[': '168', ']': '168'}),
                       '162': ('close', {'(': '163', ')': '161', '[': '168', ']': '168'}),
                       '163': ('close', {'(': '164', ')': '162', '[': '168', ']': '168'}),
                       '164': ('close', {'(': '165', ')': '163', '[': '168', ']': '168'}),
                       '165': ('close', {'(': '166', ')': '164', '[': '168', ']': '168'}),
                       '166': ('close', {'(': '167', ')': '165', '[': '168', ']': '168'}),
                       '167': ('close', {'(': '168', ')': '166', '[': '168', ']': '168'}),
                       '180': ('n', {'(': '180', ')': '180', '[': '180', ']': '180'}),
                       '170': ('close', {'(': '171', ')': '169', '[': '180', ']': '180'}),
                       '171': ('close', {'(': '172', ')': '170', '[': '180', ']': '180'}),
                       '172': ('close', {'(': '173', ')': '171', '[': '180', ']': '180'}),
                       '173': ('close', {'(': '174', ')': '172', '[': '180', ']': '180'}),
                       '174': ('close', {'(': '175', ')': '173', '[': '180', ']': '180'}),
                       '175': ('close', {'(': '176', ')': '174', '[': '180', ']': '180'}),
                       '176': ('close', {'(': '177', ')': '175', '[': '180', ']': '180'}),
                       '177': ('close', {'(': '178', ')': '176', '[': '180', ']': '180'}),
                       '178': ('close', {'(': '179', ')': '177', '[': '180', ']': '180'}),
                       '179': ('close', {'(': '180', ')': '178', '[': '180', ']': '180'}),
                       '192': ('n', {'(': '192', ')': '192', '[': '192', ']': '192'}),
                       '182': ('close', {'(': '183', ')': '181', '[': '192', ']': '192'}),
                       '183': ('close', {'(': '184', ')': '182', '[': '192', ']': '192'}),
                       '184': ('close', {'(': '185', ')': '183', '[': '192', ']': '192'}),
                       '185': ('close', {'(': '186', ')': '184', '[': '192', ']': '192'}),
                       '186': ('close', {'(': '187', ')': '185', '[': '192', ']': '192'}),
                       '187': ('close', {'(': '188', ')': '186', '[': '192', ']': '192'}),
                       '188': ('close', {'(': '189', ')': '187', '[': '192', ']': '192'}),
                       '189': ('close', {'(': '190', ')': '188', '[': '192', ']': '192'}),
                       '190': ('close', {'(': '191', ')': '189', '[': '192', ']': '192'}),
                       '191': ('close', {'(': '192', ')': '190', '[': '192', ']': '192'}),
                       '204': ('n', {'(': '204', ')': '204', '[': '204', ']': '204'}),
                       '194': ('close', {'(': '195', ')': '193', '[': '204', ']': '204'}),
                       '195': ('close', {'(': '196', ')': '194', '[': '204', ']': '204'}),
                       '196': ('close', {'(': '197', ')': '195', '[': '204', ']': '204'}),
                       '197': ('close', {'(': '198', ')': '196', '[': '204', ']': '204'}),
                       '198': ('close', {'(': '199', ')': '197', '[': '204', ']': '204'}),
                       '199': ('close', {'(': '200', ')': '198', '[': '204', ']': '204'}),
                       '200': ('close', {'(': '201', ')': '199', '[': '204', ']': '204'}),
                       '201': ('close', {'(': '202', ')': '200', '[': '204', ']': '204'}),
                       '202': ('close', {'(': '203', ')': '201', '[': '204', ']': '204'}),
                       '203': ('close', {'(': '204', ')': '202', '[': '204', ']': '204'}),
                       '216': ('n', {'(': '216', ')': '216', '[': '216', ']': '216'}),
                       '206': ('close', {'(': '207', ')': '205', '[': '216', ']': '216'}),
                       '207': ('close', {'(': '208', ')': '206', '[': '216', ']': '216'}),
                       '208': ('close', {'(': '209', ')': '207', '[': '216', ']': '216'}),
                       '209': ('close', {'(': '210', ')': '208', '[': '216', ']': '216'}),
                       '210': ('close', {'(': '211', ')': '209', '[': '216', ']': '216'}),
                       '211': ('close', {'(': '212', ')': '210', '[': '216', ']': '216'}),
                       '212': ('close', {'(': '213', ')': '211', '[': '216', ']': '216'}),
                       '213': ('close', {'(': '214', ')': '212', '[': '216', ']': '216'}),
                       '214': ('close', {'(': '215', ')': '213', '[': '216', ']': '216'}),
                       '215': ('close', {'(': '216', ')': '214', '[': '216', ']': '216'}),
                       '228': ('n', {'(': '228', ')': '228', '[': '228', ']': '228'}),
                       '218': ('close', {'(': '219', ')': '217', '[': '228', ']': '228'}),
                       '219': ('close', {'(': '220', ')': '218', '[': '228', ']': '228'}),
                       '220': ('close', {'(': '221', ')': '219', '[': '228', ']': '228'}),
                       '221': ('close', {'(': '222', ')': '220', '[': '228', ']': '228'}),
                       '222': ('close', {'(': '223', ')': '221', '[': '228', ']': '228'}),
                       '223': ('close', {'(': '224', ')': '222', '[': '228', ']': '228'}),
                       '224': ('close', {'(': '225', ')': '223', '[': '228', ']': '228'}),
                       '225': ('close', {'(': '226', ')': '224', '[': '228', ']': '228'}),
                       '226': ('close', {'(': '227', ')': '225', '[': '228', ']': '228'}),
                       '227': ('close', {'(': '228', ')': '226', '[': '228', ']': '228'}),
                       '240': ('n', {'(': '240', ')': '240', '[': '240', ']': '240'}),
                       '230': ('close', {'(': '231', ')': '229', '[': '240', ']': '240'}),
                       '231': ('close', {'(': '232', ')': '230', '[': '240', ']': '240'}),
                       '232': ('close', {'(': '233', ')': '231', '[': '240', ']': '240'}),
                       '233': ('close', {'(': '234', ')': '232', '[': '240', ']': '240'}),
                       '234': ('close', {'(': '235', ')': '233', '[': '240', ']': '240'}),
                       '235': ('close', {'(': '236', ')': '234', '[': '240', ']': '240'}),
                       '236': ('close', {'(': '237', ')': '235', '[': '240', ']': '240'}),
                       '237': ('close', {'(': '238', ')': '236', '[': '240', ']': '240'}),
                       '238': ('close', {'(': '239', ')': '237', '[': '240', ']': '240'}),
                       '239': ('f', {'(': '240', ')': '238', '[': '240', ']': '240'}),
                       '241': ('n', {'(': '241', ')': '241', '[': '241', ']': '241'})}

        blackbox = moore_from_state_setup(state_setup=state_setup)

        while make_in_complete:
            make_in_complete = False
            # paths = list(self.specification_dfa.get_all_paths(origin_state=self.specification_dfa.initial_state,
            #                                                   target_state=accepting_state))

            paths = Example.get_all_path_to_accept(dfa=self.specification_dfa)
            if len(paths) == 1:
                make_in_complete = True
                self.specification_dfa = self.create_specification_dfa(force=True)
                continue

            paths_remove = random.choices(paths, k=1)
            for p in paths_remove:
                curr_state = blackbox.initial_state
                for k in p[:-1]:
                    if k in curr_state.transitions:
                        curr_state = curr_state.transitions[k]
                    else:
                        break
                c = curr_state.transitions.copy()
                for a in c:
                    del curr_state.transitions[a]
                curr_state.output = 'sink'

            paths_left = [p for p in paths if p not in paths_remove]
            counter_not_valid = 0
            for p in paths_left:
                try:
                    blackbox.execute_sequence(origin_state=blackbox.initial_state, seq=p)
                except:
                    counter_not_valid += 1

            if counter_not_valid == len(paths):
                make_in_complete = True

        self.num_of_states_bb = len(blackbox.states)
        return blackbox

    def create_bModelCheck(self) -> bool:
        return True

    def create_bBBCheck(self) -> bool:
        return False


class ModelCheckingRandomEqualClSpecRamdomBB(Example):
    def __init__(self, bb_size=None, spec_size=None):
        self.MAX_SEQ_LEN = 20
        self.specification_dfa = None
        self.num_of_states_specification = 10 if spec_size is None else spec_size
        self.num_of_states_bb = 20 if bb_size is None else bb_size
        self.input_alphabet = self.create_input_alphabet()
        self.output_alphabet = self.create_output_alphabet()

    def create_MAX_SEQ_LEN(self) -> int:
        return self.MAX_SEQ_LEN

    def create_specification_dfa(self, force=False) -> Optional[Dfa]:
        if self.specification_dfa is None or force:

            dfa_state_setup = {}
            for i in range(self.num_of_states_specification):
                state_name = 'q' + str(i)
                next_state = 'q' + str(i + 1) if i < self.num_of_states_specification - 1 else 'q' + str(i)
                prev_state = 'q0' if i < self.num_of_states_specification - 1 else 'q' + str(i)
                action_4_next = random.choices(self.input_alphabet, k=1)
                state_setup = {}
                for action in self.input_alphabet:
                    if action in action_4_next:
                        state_setup[action] = next_state
                    else:
                        state_setup[action] = prev_state
                dfa_state_setup[state_name] = (i == (self.num_of_states_specification - 1), state_setup)

            self.specification_dfa = dfa_from_state_setup(dfa_state_setup)

        max_len = 0
        for s in self.specification_dfa.states:
            p = self.specification_dfa.get_shortest_path(
                origin_state=self.specification_dfa.initial_state, target_state=s)
            if p and len(''.join(p)) > max_len:
                max_len = len(''.join(p))

        self.MAX_SEQ_LEN = max_len * 2

        if self.input_alphabet is None:
            self.input_alphabet = self.specification_dfa.get_input_alphabet()
        return self.specification_dfa

    def create_input_alphabet(self) -> list:
        self.input_alphabet = ['0', '1']
        return self.input_alphabet

    def create_output_alphabet(self) -> list:
        self.output_alphabet = [str(unichr(97 + x)) for x in range(0, round(40 * 3 / 4))]
        return self.output_alphabet

    def create_blackbox(self, make_in_complete=True) -> MooreMachine:
        import copy

        if self.input_alphabet is None:
            self.create_input_alphabet()

        if self.output_alphabet is None:
            self.create_output_alphabet()

        # randomly generate blackbox automaton
        blackbox = generate_random_moore_machine(num_states=self.num_of_states_bb, input_alphabet=self.input_alphabet,
                                                 output_alphabet=self.output_alphabet)

        paths = Example.get_all_path_to_accept(dfa=self.specification_dfa)

        while paths is None:
            self.create_specification_dfa()

        while True:
            blackbox_copy = copy.deepcopy(blackbox)

            state_to_incomplete = random.choices(blackbox.states, k=1)[0]
            c = state_to_incomplete.transitions.copy()
            for a in c:
                del state_to_incomplete.transitions[a]
            state_to_incomplete.output = 'sink'

            counter_not_valid = 0
            for p in paths:
                try:
                    blackbox.execute_sequence(origin_state=blackbox.initial_state, seq=p)
                except:
                    counter_not_valid += 1

            if counter_not_valid == len(paths):
                if not make_in_complete:
                    print('xxxxxxx')
                    blackbox = blackbox_copy
                    break
                else:
                    blackbox = generate_random_moore_machine(num_states=self.num_of_states_bb,
                                                             input_alphabet=self.input_alphabet,
                                                             output_alphabet=self.output_alphabet)
            else:
                print('yyyyyyyy')
                make_in_complete = False

        return blackbox

    def create_bModelCheck(self) -> bool:
        return True

    def create_bBBCheck(self) -> bool:
        return False
