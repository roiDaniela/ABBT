
if __name__ == "__main__":
    import os

    rootdir = 'C:/Users/roifo/PycharmProjects/ABBT/results_ModelCheckingRandomEqualSpecBBParenthesis_100_242/'

    time_rpni = 0
    time_naive = 0
    total_seq_rpni = 0
    total_seq_naive = 0
    total_len_rpni = 0
    total_len_naive = 0
    counter = 0
    for subdir, dirs, files in os.walk(rootdir):
        for file in files:
            if file.endswith('report.txt'):
                counter+=1
                curr_time_rpni = 0
                curr_time_naive = 0
                curr_total_seq_rpni = 0
                curr_total_seq_naive = 0
                curr_total_len_rpni = 0
                curr_total_len_naive = 0
                with open(os.path.join(subdir, file)) as f:
                    try:
                        lines = [line.rstrip() for line in f]
                        time = lines[1]
                        total_seq = lines[3]
                        tota_len = lines[4]
                        spec_size = lines[5]
                        bb_size = lines[6]
                        curr_time_rpni = float(time.split()[2])
                        curr_time_naive = float(time.split()[-1])
                        curr_total_seq_rpni = float(total_seq.split()[5].replace(':',''))
                        curr_total_seq_naive = float(total_seq.split()[-1])
                        curr_total_len_rpni = float(tota_len.split()[5].replace(':',''))
                        curr_total_len_naive = float(tota_len.split()[-1])
                    except:
                        counter-=1
                    #
                time_rpni += curr_time_rpni
                time_naive += curr_time_naive
                total_seq_rpni += curr_total_seq_rpni
                total_seq_naive += curr_total_seq_naive
                total_len_rpni += curr_total_len_rpni
                total_len_naive += curr_total_len_naive

                # print(os.path.join(subdir, file))
    faster = (time_naive/counter)/(time_rpni/counter)
    print("XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX\n\n")
    print(f"faster X {faster}")
    print(f"TIME RPNI: {time_rpni/counter} TIME NAIVE: {time_naive/counter}")
    print(f"total seq RPNI: {total_seq_rpni/counter} total seq NAIVE: {total_seq_naive/counter}")
    print(f"total len RPNI: {total_len_rpni/counter} total len NAIVE: {total_len_naive/counter}")
    print(spec_size)
    print(bb_size)
    print("XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX\n\n")
