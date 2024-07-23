import argparse
import os


def parse_args():
    parser = argparse.ArgumentParser(description="Run Verse.")

    parser.add_argument('--model', nargs='?', default='asymmetric',
                        help='which model to use')

    parser.add_argument('--is_map', type=str, default='False')

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    # models = ['mdne', 'dne', 'verse', 'app']
    # models = ['mdne']
    k_numbers = ['5', '10', '20', '50']
    titles = ['5', '7', '18', '25']
    model = args.model
    if args.is_map == 'False':
        is_map = False
        print('now execute NDCG')
    if args.is_map == 'True':
        is_map = True
        print('now execute MAP')
    # is_map = args.is_map
    if not is_map:
        command_string = ''
        for title in titles:
            for k in k_numbers:
                command_string += 'nohup python3 Link-Weight-Rank.py --title ' + title + ' --is_map '+ str(is_map) +' --k ' + k + ' --model ' + model + ' &\n'
        print(command_string)
        print('\n')
        os.system(command_string)
    else:
        command_string = ''
        for title in titles:
            command_string  += 'nohup python3 Link-Weight-Rank.py  --title ' + title + ' --is_map '+ str(is_map) + ' --model ' + model + ' &\n'
        print (command_string)
        os.system(command_string)
