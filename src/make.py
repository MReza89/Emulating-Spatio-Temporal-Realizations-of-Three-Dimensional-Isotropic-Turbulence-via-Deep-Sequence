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
    file = args['file']
    gpu_ids = [','.join(str(i) for i in list(range(x, x + world_size))) for x in list(range(0, num_gpus, world_size))]
    init_seeds = [list(range(init_seed, init_seed + num_experiments, experiment_step))]
    world_size = [[world_size]]
    num_experiments = [[experiment_step]]
    resume_mode = [[resume_mode]]
    filename = '{}_{}'.format(run, file)
    data_names = [['Turb']]
    if file == 'vqvae':
        script_name = [['{}_vqvae.py'.format(run)]]
        model_names = [['vqvae']]
        control_name = [[['1', '2', '3'], ['chs-1'], ['exact-physics'], ['0.1-0.0']]]
        controls = make_controls(script_name, data_names, model_names, init_seeds, world_size, num_experiments,
                                 resume_mode, control_name)
    elif file == 'teacher':
        script_name = [['{}_convlstm.py'.format(run)]]
        model_names = [['convlstm']]
        control_name = [[['1', '2', '3'], ['chs-1'], ['exact-physics'], ['0.1-0.0'], ['3-3'], ['0']]]
        convlstm_controls = make_controls(script_name, data_names, model_names, init_seeds, world_size, num_experiments,
                                          resume_mode, control_name)
        script_name = [['{}_transformer.py'.format(run)]]
        model_names = [['transformer']]
        control_name = [[['1', '2', '3'], ['chs-1'], ['exact-physics'], ['0.1-0.0'], ['3-3'], ['0']]]
        transformer_controls = make_controls(script_name, data_names, model_names, init_seeds, world_size,
                                             num_experiments, resume_mode, control_name)
        controls = convlstm_controls + transformer_controls
    elif file == 'cyclic':
        script_name = [['{}_convlstm_cyclic.py'.format(run)]]
        model_names = [['convlstm']]
        control_name = [[['1', '2', '3'], ['chs-1'], ['exact-physics'], ['0.1-0.0'], ['3-3'], ['1']]]
        convlstm_controls = make_controls(script_name, data_names, model_names, init_seeds, world_size, num_experiments,
                                          resume_mode, control_name)
        script_name = [['{}_transformer_cyclic.py'.format(run)]]
        model_names = [['transformer']]
        control_name = [[['1', '2', '3'], ['chs-1'], ['exact-physics'], ['0.1-0.0'], ['3-3'], ['1']]]
        transformer_controls = make_controls(script_name, data_names, model_names, init_seeds, world_size,
                                             num_experiments, resume_mode, control_name)
        controls = convlstm_controls + transformer_controls
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
