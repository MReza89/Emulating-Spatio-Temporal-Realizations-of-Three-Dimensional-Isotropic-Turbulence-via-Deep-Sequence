import argparse
import os
import torch
import torch.backends.cudnn as cudnn
import models
from config import cfg
from data import fetch_dataset, make_data_loader
from utils import save, to_device, process_control, process_dataset, resume, collate, vis

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
cudnn.benchmark = True
parser = argparse.ArgumentParser(description='cfg')
for k in cfg:
    exec('parser.add_argument(\'--{0}\', default=cfg[\'{0}\'], type=type(cfg[\'{0}\']))'.format(k))
parser.add_argument('--control_name', default=None, type=str)
args = vars(parser.parse_args())
for k in cfg:
    cfg[k] = args[k]
if args['control_name']:
    cfg['control'] = {k: v for k, v in zip(cfg['control'].keys(), args['control_name'].split('_'))} \
        if args['control_name'] != 'None' else {}
cfg['control_name'] = '_'.join(
    [cfg['control'][k] for k in cfg['control'] if k not in ['seq_length', 'cyclic']]) if 'control' in cfg else ''


def main():
    process_control()
    seeds = list(range(cfg['init_seed'], cfg['init_seed'] + cfg['num_experiments']))
    for i in range(cfg['num_experiments']):
        model_tag_list = [str(seeds[i]), cfg['data_name'], cfg['model_name'], cfg['control_name']]
        cfg['model_tag'] = '_'.join([x for x in model_tag_list if x])
        print('Experiment: {}'.format(cfg['model_tag']))
        runExperiment()
    return


def runExperiment():
    seed = int(cfg['model_tag'].split('_')[0])
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    dataset = fetch_dataset(cfg['data_name'])
    process_dataset(dataset['train'])
    data_loader = make_data_loader(dataset, cfg['model_name'], {'train': False, 'test': False})
    model = eval('models.{}().to(cfg["device"])'.format(cfg['model_name']))
    load_tag = 'best'
    last_epoch, model, _, _, _ = resume(model, cfg['model_tag'], load_tag=load_tag)
    train_code = encode(data_loader['train'], model)
    test_code = encode(data_loader['test'], model)
    save(train_code, './output/code/train_{}.pt'.format(cfg['model_tag']))
    save(test_code, './output/code/test_{}.pt'.format(cfg['model_tag']))
    return


def encode(data_loader, model):
    with torch.no_grad():
        model.train(False)
        code = []
        for i, input in enumerate(data_loader):
            input = collate(input)
            input = to_device(input, cfg['device'])
            _, _, code_i = model.encode(input['uvw'])
            code.append(code_i.cpu())
            if i == 0:
                print('Code size: {}'.format(code_i.size()))
        code = torch.cat(code, dim=0)
    return code


if __name__ == "__main__":
    main()
