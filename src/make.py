import argparse
import itertools

parser = argparse.ArgumentParser(description='Config')
parser.add_argument('--run', default='train', type=str)
parser.add_argument('--num_gpus', default=4, type=int)
parser.add_argument('--world_size', default=1, type=int)
parser.add_argument('--init_seed', default=0, type=int)
parser.add_argument('--round', default=4, type=int)
parser.add_argument('--experiment_step', default=1, type=int)
parser.add_argument('--num_experiments', default=1, type=int)
parser.add_argument('--resume_mode', default=0, type=int)
parser.add_argument('--model', default=None, type=str)
parser.add_argument('--file', default=None, type=str)
args = vars(parser.parse_args())


def make_controls(script_name, data_names, model_names, init_seeds, world_size, num_experiments, resume_mode,
                  control_name):
    control_names = []
    for i in range(len(control_name)):
        control_names.extend(list('_'.join(x) for x in itertools.product(*control_name[i])))
    control_names = [control_names]
    controls = script_name + data_names + model_names + init_seeds + world_size + num_experiments + resume_mode + \
               control_names
    controls = list(itertools.product(*controls))
    return controls


def main():
    run = args['run']
    num_gpus = args['num_gpus']
    world_size = args['world_size']
    round = args['round']
    experiment_step = args['experiment_step']
    init_seed = args['init_seed']
    num_experiments = args['num_experiments']
    resume_mode = args['resume_mode']
    model = args['model']
    file = args['file'] if args['file'] is not None else model
    gpu_ids = [','.join(str(i) for i in list(range(x, x + world_size))) for x in list(range(0, num_gpus, world_size))]
    script_name = [['{}.py'.format(run)]] if run in ['encode'] else [['{}_{}.py'.format(run, file)]]
    init_seeds = [list(range(init_seed, init_seed + num_experiments, experiment_step))]
    world_size = [[world_size]]
    num_experiments = [[experiment_step]]
    resume_mode = [[resume_mode]]
    if model == 'vqvae':
        filename = '{}_{}'.format(run, model)
        model_names = [[model]]
        data_names = [['Turb']]
        control_name = [[['2', '3'], ['exact-physics'], ['0-0', '0.1-0', '0-0.0001', '0.1-0.0001']]]
        controls = make_controls(script_name, data_names, model_names, init_seeds, world_size, num_experiments,
                                 resume_mode, control_name)
    elif model in ['transformer', 'convlstm']:
        filename = '{}_{}'.format(run, model)
        model_names = [[model]]
        data_names = [['Turb']]
        control_name = [[['2', '3'], ['exact-physics'], ['0-0', '0.1-0', '0-0.0001', '0.1-0.0001'], ['4-4']]]
        controls = make_controls(script_name, data_names, model_names, init_seeds, world_size, num_experiments,
                                 resume_mode, control_name)
    else:
        raise ValueError('Not valid model')
    s = '#!/bin/bash\n'
    k = 0
    for i in range(len(controls)):
        controls[i] = list(controls[i])
        s = s + 'CUDA_VISIBLE_DEVICES=\"{}\" python {} --data_name {} --model_name {} --init_seed {} ' \
                '--world_size {} --num_experiments {} --resume_mode {} --control_name {}&\n'.format(
            gpu_ids[k % len(gpu_ids)], *controls[i])
        if k % round == round - 1:
            s = s[:-2] + '\nwait\n'
        k = k + 1
    if s[-5:-1] != 'wait':
        s = s + 'wait\n'
    print(s)
    run_file = open('./{}.sh'.format(filename), 'w')
    run_file.write(s)
    run_file.close()
    return


if __name__ == '__main__':
    main()
