import argparse
import datetime
import models
import os
import shutil
import time
import torch
import torch.backends.cudnn as cudnn
from config import cfg
from data import BatchDataset
from metrics import Metric
from utils import save, load, to_device, process_control, process_dataset, make_optimizer, make_scheduler, resume
from logger import Logger

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
    [cfg['control'][k] for k in cfg['control'] if cfg['control'][k]]) if 'control' in cfg else ''
cfg['ae_name'] = 'vqvae'


def main():
    process_control()
    seeds = list(range(cfg['init_seed'], cfg['init_seed'] + cfg['num_experiments']))
    for i in range(cfg['num_experiments']):
        cfg['ae_control_name'] = '_'.join([cfg['control'][k] for k in cfg['control'] if k not in ['seq_length']])
        ae_tag_list = [str(seeds[i]), cfg['data_name'], cfg['ae_name'], cfg['ae_control_name']]
        model_tag_list = [str(seeds[i]), cfg['data_name'], \
                          cfg['model_name'] + '_cyclic' if cfg['cyclic_train']==1 else cfg['model_name'], cfg['control_name']]
        cfg['ae_tag'] = '_'.join([x for x in ae_tag_list if x])
        cfg['model_tag'] = '_'.join([x for x in model_tag_list if x])
        print('Experiment: {}'.format(cfg['model_tag']))
        runExperiment()
    return


def runExperiment():
    seed = int(cfg['model_tag'].split('_')[0])
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    dataset = {}
    dataset['train'] = load('./output/code/train_{}.pt'.format(cfg['ae_tag']))
    dataset['test'] = load('./output/code/test_{}.pt'.format(cfg['ae_tag']))
    process_dataset(dataset)
    ae = eval('models.{}().to(cfg["device"])'.format(cfg['ae_name']))
    _, ae, _, _, _ = resume(ae, cfg['ae_tag'], load_tag='best')
    model = eval('models.{}().to(cfg["device"])'.format(cfg['model_name']))
    optimizer = make_optimizer(model, cfg['model_name'])
    scheduler = make_scheduler(optimizer, cfg['model_name'])
    metric = Metric({'train': ['Loss'], 'test': ['Loss', 'MSE']})
    if cfg['resume_mode'] == 1:
        last_epoch, model, optimizer, scheduler, logger = resume(model, cfg['model_tag'], optimizer, scheduler)
    else:
        last_epoch = 1
        current_time = datetime.datetime.now().strftime('%b%d_%H-%M-%S')
        logger_path = 'output/runs/train_{}_{}'.format(cfg['model_tag'], current_time)
        logger = Logger(logger_path)
    if cfg['world_size'] > 1:
        model = torch.nn.DataParallel(model, device_ids=list(range(cfg['world_size'])))
    for epoch in range(last_epoch, cfg[cfg['model_name']]['num_epochs'] + 1):
        logger.safe(True)
        train(dataset['train'], model, optimizer, metric, logger, epoch)
        test(dataset['test'], ae, model, metric, logger, epoch)
        if cfg[cfg['model_name']]['scheduler_name'] == 'ReduceLROnPlateau':
            scheduler.step(metrics=logger.mean['train/{}'.format(metric.pivot_name)])
        else:
            scheduler.step()
        logger.safe(False)
        model_state_dict = model.module.state_dict() if cfg['world_size'] > 1 else model.state_dict()
        save_result = {
            'cfg': cfg, 'epoch': epoch + 1, 'model_dict': model_state_dict,
            'optimizer_dict': optimizer.state_dict(), 'scheduler_dict': scheduler.state_dict(),
            'logger': logger}
        save(save_result, './output/model/{}_checkpoint.pt'.format(cfg['model_tag']))
        if metric.compare(logger.mean['test/{}'.format(metric.pivot_name)]):
            metric.update(logger.mean['test/{}'.format(metric.pivot_name)])
            shutil.copy('./output/model/{}_checkpoint.pt'.format(cfg['model_tag']),
                        './output/model/{}_best.pt'.format(cfg['model_tag']))
        logger.reset()
    logger.safe(False)
    return


def train(dataset, model, optimizer, metric, logger, epoch):
    model.train(True)
    dataset = BatchDataset(dataset, cfg['seq_length'])
    start_time = time.time()
    for i, input in enumerate(dataset):
        if cfg['cyclic_train'] == 1 and i != 0:
            input['code'] = output['ncode']
        input_size = input['code'].size(0)
        input = to_device(input, cfg['device'])
        optimizer.zero_grad()
        output = model(input)
        output['loss'] = output['loss'].mean() if cfg['world_size'] > 1 else output['loss']
        output['loss'].backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
        optimizer.step()
        evaluation = metric.evaluate(metric.metric_name['train'], input, output)
        logger.append(evaluation, 'train', n=input_size)
        if i % int((len(dataset) * cfg['log_interval']) + 1) == 0:
            batch_time = (time.time() - start_time) / (i + 1)
            lr = optimizer.param_groups[0]['lr']
            epoch_finished_time = datetime.timedelta(seconds=round(batch_time * (len(dataset) - i - 1)))
            exp_finished_time = epoch_finished_time + datetime.timedelta(
                seconds=round((cfg[cfg['model_name']]['num_epochs'] - epoch) * batch_time * len(dataset)))
            info = {'info': ['Model: {}'.format(cfg['model_tag']),
                             'Train Epoch: {}({:.0f}%)'.format(epoch, 100. * i / len(dataset)),
                             'Learning rate: {:.6f}'.format(lr), 'Epoch Finished Time: {}'.format(epoch_finished_time),
                             'Experiment Finished Time: {}'.format(exp_finished_time)]}
            logger.append(info, 'train', mean=False)
            print(logger.write('train', metric.metric_name['train']))
    return


def test(dataset, ae, model, metric, logger, epoch):
    with torch.no_grad():
        model.train(False)
        ae.train(False)
        dataset = BatchDataset(dataset, cfg['seq_length'])
        for i, input in enumerate(dataset):
            input_size = input['code'].size(0)
            input = to_device(input, cfg['device'])
            output = model(input)
            input['uvw'] = ae.decode_code(input['ncode'].view(-1, *input['ncode'].size()[2:]))
            output['uvw'] = ae.decode_code(output['ncode'].view(-1, *output['ncode'].size()[2:]))
            output['loss'] = output['loss'].mean() if cfg['world_size'] > 1 else output['loss']
            evaluation = metric.evaluate(metric.metric_name['test'], input, output)
            logger.append(evaluation, 'test', input_size)
        info = {'info': ['Model: {}'.format(cfg['model_tag']), 'Test Epoch: {}({:.0f}%)'.format(epoch, 100.)]}
        logger.append(info, 'test', mean=False)
        print(logger.write('test', metric.metric_name['test']))
    return


if __name__ == "__main__":
    main()
