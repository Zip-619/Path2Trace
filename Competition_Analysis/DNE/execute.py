import argparse
import os


def parse_args():
    parser = argparse.ArgumentParser(description="Run Verse.")

    parser.add_argument('--dim', type=int, default=0,
                        help='Number of dimensions. Default is 64.')

    parser.add_argument('--model', nargs='?', default='mdne',
                        help='model name.')

    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    dim = args.dim
    # jrates = [0.05, 0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85, 0.95]
    jrates = [0.001, 0.1, 0.9, 0.999]
    command = ''
    if args.model == 'mdne':
        for jrate in jrates:
            command += 'nohup python3 parameter-sensitively-MDNE.py --dimensions ' + str(dim) + ' --jrate ' + str(jrate) + ' &\n'
        print(command)
        os.system(command)

    if args.model == 'dne':
        for jrate in jrates:
            command += 'nohup python3 parameter-sensitively-DNE.py --dimensions ' + str(dim) + ' --jrate ' + str(jrate) + ' &\n'
        print(command)
        os.system(command)

    # nsamples = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    # command = ''
    # if args.model == 'mdne':
    #     for nsample in nsamples:
    #         command += 'nohup python3 parameter-sensitively-MDNE.py --nsamples ' + str(nsample) + ' &\n'
    #     print(command)
    #     os.system(command)
    #
    #
    # if args.model == 'dne':
    #     for nsample in nsamples:
    #         command += 'nohup python3 parameter-sensitively-DNE.py --nsamples ' + str(nsample) + ' &\n'
    #     print(command)
    #     os.system(command)