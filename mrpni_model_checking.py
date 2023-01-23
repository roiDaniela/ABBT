import copy
import random
import time
import uuid
from collections import defaultdict
from pathlib import Path

import networkx as nx
from aalpy.automata import Dfa, DfaState, MooreMachine
from aalpy.learning_algs import run_RPNI
from aalpy.utils import dfa_from_state_setup, generate_random_dfa, generate_random_moore_machine
from termcolor import colored

import Examples

bPrint = False
MAX_SEQ_LEN = 10
used_sequences = defaultdict(bool)
DATA = set()
used_output = defaultdict(int)
M = defaultdict(set)
l_t = defaultdict(set)
KERNEL = defaultdict()
OUT = []
MIN_OUTPUT = ()
# choices = defaultdict(list)
RPNI_DONE_COUNTER = 0
BLACKLIST_SPEC = defaultdict(bool)
FIRST_SEQ_SUCC = 0

results_wanted_good = 10
results_wanted_bad = 5
max_faster_good = 1
max_faster_bad = 1
time_rpni = 0
rpni_model = None


def main():
    global results_wanted_good
    global results_wanted_bad
    global max_faster_good
    global max_faster_bad

    # choose example
    # example = Examples.ModelCheckingRandomBigSpecBBDivisibility(spec_size=20)

    while True:
        example = random.choice([
            Examples.ModelCheckingRandomEqualSpecBBParenthesis(),
            # Examples.ModelCheckingRandomEqualClSpecRamdomBB(bb_size=50, spec_size=20),
            # Examples.ModelCheckingRandomEqualClSpecRamdomBB(bb_size=70, spec_size=30)
            # Examples.ModelCheckingRandomBigSpecBBDivisibility(spec_size=100),
            # Examples.ModelCheckingRandomBigSpecBBDivisibility(spec_size=50),
            # Examples.ModelCheckingRandomBigSpecBBDivisibility(spec_size=10),
            # Examples.Divisibility(),
            # Examples.Parenthesis(),
            # Examples.SuffixOne(),
            # Examples.ZeroOneTwo(),
            # Examples.Substrings(),
            # Examples.EqualStub()
        ]
        )
        run_iterations(example=example)
        # break


def run_iterations(example):
    global results_wanted_good
    global results_wanted_bad
    global max_faster_good
    global max_faster_bad
    global M
    global l_t
    global KERNEL
    global OUT
    global MIN_OUTPUT
    global used_sequences
    global used_output
    global DATA
    global MAX_SEQ_LEN
    global RPNI_DONE_COUNTER
    global TOTAL_RPNI_DATA
    # global choices
    global BLACKLIST_SPEC
    global bPrint
    global time_rpni
    global FIRST_SEQ_SUCC

    first_try = True
    FIRST_SEQ_SUCC = 0
    ITERATIONS_NUM = 10

    rpni_succ = False
    total_time_naive = 0
    total_time_rpni = 0
    num_sequences_rpni = 0
    len_sequences_rpni = 0
    num_sequences_naive = 0
    len_sequences_naive = 0

    specification_dfa = example.create_specification_dfa()
    input_alphabet = example.create_input_alphabet()
    MAX_SEQ_LEN = example.create_MAX_SEQ_LEN()
    blackbox = example.create_blackbox(make_in_complete=True)

    # setting validation
    if specification_dfa is not None:
        assert blackbox.get_input_alphabet() == specification_dfa.get_input_alphabet()
        assert any([s.is_accepting for s in specification_dfa.states])
    assert all([len(x) == 1 for x in blackbox.get_input_alphabet()])
    ##

    folder_name_results = 'results_' + str(
        type(example).__name__) + '' if example.create_bBBCheck() else 'results_' + str(
        type(example).__name__) + f'_{specification_dfa.size}' + f'_{blackbox.size}'

    while not rpni_succ:
        if not first_try:
            # when blackbox is randomly generated and there is error in setting we are here.
            blackbox = example.create_blackbox(make_in_complete=True)
        first_try = False
        if bPrint:
            print(colored('fresh start', 'blue'))

        specification_dfa = example.specification_dfa
        MAX_SEQ_LEN = example.MAX_SEQ_LEN

        # create blacklist of specification states
        if specification_dfa is not None:
            blacklist = Examples.Example.get_blacklist_states_4_spec(specification_dfa=specification_dfa)
            for s in blacklist:
                BLACKLIST_SPEC[s] = True

        bModelCheck = example.create_bModelCheck()
        bBBCheck = example.create_bBBCheck()

        ##
        total_time_naive = 0
        total_time_rpni = 0
        num_sequences_rpni = 0
        len_sequences_rpni = 0
        num_sequences_naive = 0
        RPNI_DONE_COUNTER = 0
        TOTAL_RPNI_DATA = list()

        for iteration_n in range(ITERATIONS_NUM):
            # init globals
            used_sequences = defaultdict(bool)
            DATA = set()
            used_output = defaultdict(int)
            M = defaultdict(set)
            KERNEL = defaultdict()
            OUT = []
            MIN_OUTPUT = ()
            BLACKLIST_SPEC = defaultdict(bool)
            bPrint = False
            time_rpni = time.time()

            # prepare for rpni first build
            if specification_dfa is not None:
                M[''].add(specification_dfa.initial_state)

            found_err = False

            while not len([v for v in used_output.values() if v > 0]) > 0.5 * (1 + len(blackbox.get_output_alphabet())):
                found_new_output, found_err = generate_seq_bb(blackbox=blackbox, prefix=None, bModelCheck=bModelCheck,
                                                              bBBCheck=bBBCheck,
                                                              bExitFoundOutput=False, input_al=input_alphabet,
                                                              specification_dfa=specification_dfa)
                if found_err:
                    rpni_succ = True
                    break

            # mrpni
            if not found_err:
                rpni_succ, found_err = rpni_and_expand(blackbox=blackbox, specification_dfa=specification_dfa,
                                                       input_alphabet=input_alphabet)

                if not rpni_succ:
                    if bPrint:
                        print(colored('wrong setting', 'red'))
                    break

            # extend kernel states
            if not found_err:
                timeout = time.time() + 60 * 10  # 10 minutes from now

                while True:
                    if time.time() > timeout:
                        return False, False

                    eps = random.uniform(0, 1)
                    found_new_output, found_err = False, False

                    if eps < 0.2 or len(OUT) == 0:
                        found_new_output, found_err = generate_seq_bb(blackbox=blackbox, prefix=None,
                                                                      bModelCheck=bModelCheck, bBBCheck=bBBCheck,
                                                                      bExitFoundOutput=True, input_al=input_alphabet,
                                                                      specification_dfa=specification_dfa)
                    else:
                        if eps >= 0.2:
                            if bPrint:
                                print(colored('exploring out states after rpni', 'blue'))

                            # pickup an incomplete state
                            o = random.choices(OUT, weights=[1 + len(M[o.state_id]) for o in OUT], k=1)[0]
                            # o = random.choice(OUT)
                            found_new_output, found_err = generate_seq_bb(blackbox=blackbox, prefix=o,
                                                                          bModelCheck=bModelCheck, bBBCheck=bBBCheck,
                                                                          bExitFoundOutput=True,
                                                                          input_al=input_alphabet,
                                                                          specification_dfa=specification_dfa)

                    if found_err:
                        break

                    if found_new_output or (
                            used_output[MIN_OUTPUT[0][0]] > MIN_OUTPUT[0][1] + (1 + MIN_OUTPUT[0][1] / MIN_OUTPUT[1]) *
                            MIN_OUTPUT[0][1]):
                        rpni_succ, found_err = rpni_and_expand(blackbox=blackbox, specification_dfa=specification_dfa,
                                                               input_alphabet=input_alphabet)
                        if not rpni_succ:
                            if bPrint:
                                print(colored('wrong setting', 'red'))
                            break

                        if found_err:
                            break

            if not rpni_succ:
                if bPrint:
                    print(colored('wrong setting', 'red'))
                break

            total_time_rpni += time.time() - time_rpni
            num_sequences_rpni += len([v for v in used_sequences.values() if v])
            len_sequences_rpni += sum([len(k) for k, v in used_sequences.items() if v])

            # naive random
            time_naive = time.time()

            bPrint = False
            used_sequences = defaultdict(bool)
            M = defaultdict(set)
            if specification_dfa is not None:
                M[''].add(specification_dfa.initial_state)

            found_err = False
            while not found_err:
                if bPrint:
                    print(colored(f'generating from initial {blackbox.initial_state.state_id}', 'blue'))

                seq = ''
                while len(seq) < MAX_SEQ_LEN:

                    seq += ''.join(random.choices(blackbox.get_input_alphabet(), k=1))
                    used_sequences[seq] = True

                    try:
                        output = blackbox.compute_output_seq(blackbox.initial_state, seq)[-1]
                        if bBBCheck:
                            if bPrint:
                                print(colored(f'seq {seq}', 'magenta'))
                            if output == 'f':
                                found_err = True
                                break
                        elif bModelCheck:
                            if bPrint:
                                print(colored(f'seq {seq}', 'magenta'))
                            if specification_dfa.execute_sequence(origin_state=specification_dfa.initial_state,
                                                                  seq=seq):
                                curr_state = specification_dfa.initial_state
                                for a in seq:
                                    curr_state = curr_state.transitions[a]

                                if curr_state.is_accepting:
                                    found_err = True
                                    break
                                else:
                                    print(colored(f'checked {seq}', 'red'))
                        else:
                            print(colored(f'checked {seq}', 'red'))
                            found_err = False
                    except:
                        if bPrint:
                            print(colored(f'checked {seq}', 'yellow'))
                        seq = seq[:-1] + ''.join(random.choices(blackbox.get_input_alphabet(), k=1))
                        found_err = False
                        if bPrint:
                            print(colored(f'generate another char{seq}', 'yellow'))

                if found_err:
                    if bPrint:
                        print(colored(f'found error from initial {blackbox.initial_state.state_id}', 'green'))
                    rpni_succ = True
                    break
                if bPrint:
                    print(
                        colored(f'still not found continue generating from initial {blackbox.initial_state.state_id} ',
                                'blue'))

            num_sequences_naive += len([v for v in used_sequences.values() if v])
            len_sequences_naive += sum([len(k) for k, v in used_sequences.items() if v])

            total_time_naive += time.time() - time_naive

            print(f'iteration num {iteration_n}')

    avg_time_rpni = total_time_rpni / ITERATIONS_NUM
    avg_time_naive = total_time_naive / ITERATIONS_NUM
    avg_num_rpni = num_sequences_rpni / ITERATIONS_NUM
    avg_num_naive = num_sequences_naive / ITERATIONS_NUM
    avg_len_rpni = len_sequences_rpni / ITERATIONS_NUM
    avg_len_naive = len_sequences_naive / ITERATIONS_NUM

    print(colored(
        f'TIME RPNI: {avg_time_rpni} succ first time {FIRST_SEQ_SUCC / ITERATIONS_NUM} spec size {0 if specification_dfa is None else specification_dfa.size} bb size {blackbox.size} TOTAL seq tested {avg_num_rpni} len seq {avg_len_rpni}',
        'green'))

    print(colored(
        f'TIME NAIVE: {avg_time_naive} spec size {0 if specification_dfa is None else specification_dfa.size} bb size {blackbox.size} TOTAL seq tested {avg_num_naive} len seq {avg_len_naive}',
        'green'))

    id_report = str(uuid.uuid4())

    bad_good = ''

    BAD_GOOD_RATIO = 1

    if avg_time_rpni > 0 and avg_time_rpni * BAD_GOOD_RATIO < avg_time_naive and avg_num_rpni < avg_num_naive:
        print(colored('success!', 'green'))
        bad_good = 'good'
        results_wanted_good -= 1
    elif avg_time_naive > 0 and avg_time_naive * BAD_GOOD_RATIO < avg_time_rpni:
        print(colored('FAIL!', 'red'))
        bad_good = 'bad'
        results_wanted_bad -= 1
    else:
        print(colored('FAIL!', 'red'))

    if bad_good != '':
        rpni_done_avg = RPNI_DONE_COUNTER / ITERATIONS_NUM
        rpni_data_avg = sum([len(d) for d in TOTAL_RPNI_DATA]) / ITERATIONS_NUM
        rpni_data_len_avg = sum([sum([len(x[0]) for x in d]) for d in TOTAL_RPNI_DATA]) / ITERATIONS_NUM
        Path(f"{folder_name_results}/{bad_good}/{id_report}").mkdir(parents=True, exist_ok=True)
        s = prepare_report(avg_num_rpni=avg_num_rpni, avg_len_rpni=avg_len_rpni, avg_num_naive=avg_num_naive,
                           avg_len_naive=avg_len_naive, avg_time_naive=avg_time_naive,
                           avg_time_rpni=avg_time_rpni, blackbox=blackbox, specification_dfa=specification_dfa,
                           bad_good=bad_good, rpni_done_avg=rpni_done_avg, rpni_data_avg=rpni_data_avg,
                           rpni_data_len_avg=rpni_data_len_avg)
        f = open(f"{folder_name_results}/{bad_good}/{id_report}/report.txt", "w")
        f.write(s)
        f.close()
        # if specification_dfa is not None:
        #     specification_dfa.visualize(path=f'{folder_name_results}/{bad_good}/{id_report}/specification', bOpenBrowser=False)
        # blackbox.visualize(path=f'{folder_name_results}/{bad_good}/{id_report}/blackbox', bOpenBrowser=False)


def prepare_report(avg_time_rpni, avg_time_naive, avg_num_rpni, avg_len_rpni, avg_num_naive, avg_len_naive, blackbox,
                   specification_dfa, bad_good, rpni_done_avg, rpni_data_avg, rpni_data_len_avg):
    global max_faster_good
    global max_faster_bad

    all_possible_paths_specification = None
    possible_path_in_bb = None
    shortest_valid_path = ['']

    if specification_dfa is not None and specification_dfa.size < 30:
        try:
            all_possible_paths_specification = Examples.Example.get_all_path_to_accept(dfa=specification_dfa)
            possible_path_in_bb = Examples.Example.validate_paths_in_blackbox(blackbox=blackbox,
                                                                              paths=all_possible_paths_specification)

            blackbox_as_dfa = dfa_from_state_setup(blackbox.convert_2_complete_dfa_setup())
            intersect_dfa = dfa_from_state_setup(specification_dfa.intersection_setup(D2=blackbox_as_dfa))
            shortest_valid_path = Examples.Example.get_shortest_path_to_accept(intersect_dfa)
        except:
            print()

    if bad_good == 'bad':
        x = avg_time_rpni / avg_time_naive if avg_time_naive > 0 else 1
        if x > max_faster_bad:
            max_faster_bad = x
    else:
        x = avg_time_naive / avg_time_rpni if avg_time_rpni > 0 else 1
        if x > max_faster_good:
            max_faster_good = x

    time = f'faster x {x}\n' \
           f'TIME RPNI: {avg_time_rpni} : TIME NAIVE: {avg_time_naive}\n ' \
           f'TIME RPNI: {round(avg_time_rpni, 4)} : TIME NAIVE: {round(avg_time_naive, 4)}\n ' \
           f'TOTAL seq tested RPNI : {avg_num_rpni}: TOTAL seq tested NAIVE {avg_num_naive}\n ' \
           f'len seq tested RPNI : {avg_len_rpni}: TOTAL seq tested NAIVE {avg_len_naive}\n ' \
           f'spec size {specification_dfa.size if specification_dfa is not None else 0} \n ' \
           f'bb size {blackbox.size} \n ' \
           f'outputs blackbox {blackbox.get_output_alphabet()}\n ' \
           f'input {blackbox.get_input_alphabet()}\n' \
           f'RPNI DONE AVG: {rpni_done_avg}\n' \
           f'RPNI DATA SIZE AVG : {rpni_data_avg}\n' \
           f'RPNI DATA LEN AVG : {rpni_data_len_avg}\n'

    paths_s = '\n\n***************\n\n'

    paths_s += '\n'.join(shortest_valid_path)

    paths_s = '\n\n***************\n\n'

    if possible_path_in_bb is not None:
        paths_s += '\n'.join(possible_path_in_bb)

    paths_s += '\n\n***************\n\n'

    if all_possible_paths_specification is not None:
        paths_s += '\n'.join(all_possible_paths_specification)

    return time + paths_s


def rpni_and_expand(input_alphabet, blackbox=None, specification_dfa=None):
    global M
    global l_t
    global KERNEL
    global OUT
    global MIN_OUTPUT
    global used_sequences
    global used_output
    global DATA
    global RPNI_DONE_COUNTER
    global TOTAL_RPNI_DATA
    global rpni_model
    global FIRST_SEQ_SUCC
    global time_rpni

    M = defaultdict(set)
    KERNEL = defaultdict(bool)
    OUT = []

    rpni_model = run_RPNI(list(DATA), automaton_type='moore', print_info=False, input_alphabet=input_alphabet,
                          bKernel=True)

    RPNI_DONE_COUNTER += 1
    TOTAL_RPNI_DATA.append(DATA)
    if rpni_model is None:
        return False, False

    update_kernel_out_states(model=rpni_model, input_al=input_alphabet)

    # expand
    if specification_dfa is not None:
        M[rpni_model.initial_state.state_id].add(specification_dfa.initial_state)
        q_prime, s_prime = expand(s=rpni_model.initial_state, input_alphabet=input_alphabet,
                                  q=specification_dfa.initial_state)

        if q_prime is not None:
            found_err = unexpand_and_validate(first_out_kernel_state=s_prime, input_al=input_alphabet, sub_seq='',
                                              blackbox=blackbox, specification_dfa=specification_dfa,
                                              accepting_q=q_prime)

            if found_err:
                if bPrint:
                    print(colored(f'error found during building kernel {found_err}', 'green'))

                FIRST_SEQ_SUCC += 1
                return True, True

        # expand for first out level of model
        for o in OUT:
            alpha = o.state_id[-1]
            sub_set = set(q.transitions[alpha] for q in M[o.state_id[:-1]] if not BLACKLIST_SPEC[q])
            M[o.state_id] = M[o.state_id].union(sub_set)

    # for naive setting
    elif blackbox is not None:
        for s in rpni_model.states:
            try:
                output = blackbox.compute_output_seq(blackbox.initial_state, s.state_id)[-1]
            except:
                return False, False
            if output == 'f':
                if bPrint:
                    print(colored(f'error found during building kernel {s.state_id}', 'green'))
                return True, True

    MIN_OUTPUT = min(used_output.items(), key=lambda x: x[1]), sum(used_output.values())

    return True, False


def update_kernel_out_states(model, input_al):
    global M
    global l_t
    global KERNEL
    global OUT
    global MIN_OUTPUT
    global used_sequences
    global used_output
    global DATA

    KERNEL[model.initial_state.state_id] = model.initial_state
    for s in model.states:
        if len(s.transitions) == len(input_al):
            KERNEL[s.state_id] = s
        elif s != model.initial_state and any(len(x.transitions.keys()) > 0 for x in s.transitions.values()):
            OUT.append(s)

    # filter only first level
    OUT = [x for x in OUT if KERNEL[x.state_id[:-1]]]

    OUT.sort(key=lambda x: len(x.state_id))


def generate_seq_bb(blackbox: MooreMachine, prefix=None, bBBCheck=False, bModelCheck=False, bExitFoundOutput=False,
                    input_al=None, specification_dfa=None, naive_random=False) -> (
        bool, bool):
    global M
    global l_t
    global KERNEL
    global OUT
    global MIN_OUTPUT
    global used_sequences
    global used_output
    global DATA
    global time_rpni

    if input_al is None:
        input_al = blackbox.get_input_alphabet()

    found_new_output = False
    l = 0
    prev_seq = ''
    if prefix is None:
        start_state = blackbox.initial_state
    else:
        l = len(prefix.state_id)
        prev_seq = prefix.state_id
        start_state = blackbox.initial_state
        for action in prev_seq:
            if action not in start_state.transitions:
                return found_new_output, False
            start_state = start_state.transitions[action]

    mex_sub_seq_len = MAX_SEQ_LEN - l

    seq = ''.join(random.choices(input_al, k=mex_sub_seq_len))

    if used_sequences[prev_seq + seq] and not naive_random:
        return found_new_output, False

    for j in range(1, len(seq) + 1):
        sub_seq = seq[:j]

        output = None
        options = copy.copy(input_al)
        while output is None:
            try:
                output = blackbox.compute_output_seq(start_state, prev_seq + sub_seq)[-1]
            except:
                try:
                    options.remove(str(sub_seq[-1]))
                    if len(options) == 0:
                        return found_new_output, False

                    sub_seq = sub_seq[:j - 1] + str(random.choice(options))
                except:
                    return found_new_output, False

        # stop by model checking with specification
        if bModelCheck:
            kernel_subseq = prev_seq + sub_seq[:-1]
            sub_set_spec_last = M[kernel_subseq]
            sub_set_spec = set([q.transitions[sub_seq[-1]] for q in sub_set_spec_last if not BLACKLIST_SPEC[q]])
            M[prev_seq + sub_seq] = M[prev_seq + sub_seq].union(sub_set_spec)
            used_sequences[prev_seq + sub_seq] = True

            for q in sub_set_spec:
                if q.is_accepting and start_state != blackbox.initial_state:
                    # validation
                    found_err = unexpand_and_validate(first_out_kernel_state=prefix, accepting_q=q, sub_seq=sub_seq,
                                                      specification_dfa=specification_dfa, blackbox=blackbox,
                                                      input_al=input_al)

                    if found_err and bPrint:
                        print(colored(f'error found from kernel state {prev_seq} actions {sub_seq}', 'green'))

                    return found_new_output, found_err
                else:
                    if bPrint:
                        print(colored(f'checked {prev_seq + sub_seq}', 'red'))

        # naive setting, stop by fail state in blackbox
        if bBBCheck:
            used_sequences[prev_seq + sub_seq] = True
            if output == 'f':
                if bPrint:
                    print(colored(f'error found from kernel state {prev_seq} actions {sub_seq}', 'green'))
                return found_new_output, True
            else:
                if bPrint:
                    print(colored(f'checked {prev_seq + sub_seq}', 'red'))

        used_output[output] += 1
        if used_output[output] == 1:
            found_new_output = True
            if not bBBCheck and not bModelCheck and bExitFoundOutput:
                return found_new_output, False

        x = used_output[output] / sum(used_output.values())
        if used_output[output] < max(5, int(used_output[output] * (1 - x) / 4)):
            DATA.add((prev_seq + sub_seq, output))

    return found_new_output, False


def unexpand_and_validate(first_out_kernel_state, accepting_q, sub_seq, blackbox, specification_dfa, input_al) -> bool:
    global M
    global l_t
    global KERNEL
    global OUT
    global MIN_OUTPUT
    global used_sequences
    global used_output
    global DATA
    global RPNI_DONE_COUNTER
    global TOTAL_RPNI_DATA
    global rpni_model
    global time_rpni
    global FIRST_SEQ_SUCC

    if first_out_kernel_state is None:
        return True

    # check if current sequence leads to error
    prefix = first_out_kernel_state.state_id
    output = None
    try:
        output = blackbox.compute_output_seq(blackbox.initial_state, prefix + sub_seq)[-1]
    except:
        output = None

    if output is not None:
        curr_state = specification_dfa.initial_state
        for a in prefix + sub_seq:
            curr_state = curr_state.transitions[a]
            if curr_state.is_accepting:
                FIRST_SEQ_SUCC += 1
                return True

    # alternative way
    for q in M[first_out_kernel_state.state_id]:
        curr_q = copy.copy(q)
        for a in sub_seq:
            if a not in curr_q.transitions.keys():
                break
            curr_q = curr_q.transitions[a]
        if curr_q.is_accepting:
            l_t[first_out_kernel_state.state_id].add(q)

    kernel_and_out = list([k for k in KERNEL.values() if k != False]) + [first_out_kernel_state]
    G = Examples.Example.to_digraph_cross_product(kernel_and_out, specification_dfa.states)

    all_accepting_options = [first_out_kernel_state.state_id + q_.state_id for q_ in
                             l_t[first_out_kernel_state.state_id]]

    for q_product in all_accepting_options:
        try:
            path = nx.shortest_path(G, source=specification_dfa.initial_state.state_id, target=q_product)
            edge_labels = nx.get_edge_attributes(G, 'label')
            edge_path = [(path[i], path[i + 1]) for i in range(len(path) - 1)]
            edge_labels_path = ''.join([edge_labels[edge] for edge in edge_path])

            output = None
            try:
                output = blackbox.compute_output_seq(blackbox.initial_state, edge_labels_path + sub_seq)[-1]
            except:
                output = None

            if output is not None:
                return True
        except nx.NetworkXNoPath:
            continue

    return False


def expand(s, q, input_alphabet: list):
    global M
    global l_t
    global KERNEL
    global OUT
    global MIN_OUTPUT
    global used_sequences
    global used_output
    global DATA

    if q.is_accepting:
        return s.state_id

    for alpha in input_alphabet:
        if alpha in s.transitions.keys() and alpha in q.transitions.keys() and s.transitions[
            alpha].state_id in KERNEL.keys():
            s_prime = s.transitions[alpha]
            q_prime = q.transitions[alpha]

            if q_prime.is_accepting:
                return q_prime, s_prime

            if q_prime not in M[s_prime.state_id] and not BLACKLIST_SPEC[q_prime]:
                M[s_prime.state_id].add(q_prime)
                return expand(s=s_prime, q=q_prime, input_alphabet=input_alphabet)

    return None, None


if __name__ == "__main__":
    main()
