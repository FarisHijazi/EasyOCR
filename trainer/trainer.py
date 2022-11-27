import os
import wandb
import dotenv

dotenv.load_dotenv("~/.env") # activate base env
dotenv.load_dotenv(".env.secrets") # override specific project env

# os.environ['WANDB_NOTEBOOK_NAME'] = "trainer.ipynb"
os.environ['WANDB_PROJECT'] = "easyocr-ehkaam-nid-numbers"
os.environ['WANDB_BASE_URL'] = "http://localhost:8080" # using local wandb server
# os.environ['WANDB_MODE'] = 'offline'

import torch.backends.cudnn as cudnn
import yaml
from train import train
from utils import AttrDict
import pandas as pd
import glob
import argparse


def get_config(file_path):
    with open(file_path, 'r', encoding="utf8") as stream:
        opt = yaml.safe_load(stream)
    opt = AttrDict(opt)
    if opt.lang_char == 'None':
        characters = ''
        for data in opt['select_data'].split('-'):
            csv_path = os.path.join(opt['train_data'], data, 'labels.csv')
            df = pd.read_csv(csv_path, sep='^([^,]+),', engine='python', usecols=['filename', 'words'], keep_default_na=False, dtype=str)
            all_char = ''.join(df['words'])
            characters += ''.join(set(all_char))
        characters = sorted(set(characters))
        opt.character= ''.join(characters)
    elif 'character' not in opt:
        opt.character = opt.number + opt.symbol + opt.lang_char
    print('character:', len(opt.character), opt.character)
    safety_checks(opt, file_path)
    os.makedirs(f'./saved_models/{opt.experiment_name}', exist_ok=True)
    return opt

def get_fname(fpath):
    # returns filename without extension
    return '.'.join(os.path.basename(fpath.rstrip('/').rstrip('\\')).split('.')[:-1])


def safety_checks(opt, config_path):
    intersection = set(opt['number']) & set(opt['lang_char'])
    assert len(intersection) == 0, f'number and lang_char should not have common characters: {intersection}'
    intersection = set(opt['symbol']) & set(opt['lang_char'])
    assert len(intersection) == 0, f'symbol and lang_char should not have common characters: {intersection}'
    intersection = set(opt['symbol']) & set(opt['number'])
    assert len(intersection) == 0, f'symbol and number should not have common characters: {intersection}'

    key_blacklist = ['number', 'symbol', 'lang_char', 'character']
    for k in opt:
        if type(opt[k]) == str and k not in key_blacklist:
            opt[k] = opt[k].format(
                config_path=config_path,
                config_name=get_fname(config_path),
                **opt
            )

    opt['wandb_kwargs'] = opt.get('wandb_kwargs', {})
    try:
        opt['saved_model'] = sorted(glob.glob(opt['saved_model']), key=lambda t: os.stat(t).st_mtime)[-1]
        print('saved_model', opt['saved_model'])
        opt['wandb_kwargs']['resume'] = True
    except Exception as e:
        print(e, 'Failed to find saved_model from', opt['saved_model'], 'falling back to base_model: ', opt['base_model'], end='')
        opt['saved_model'] = opt['base_model']
        assert os.path.isfile(opt['base_model'])
        opt['wandb_kwargs']['resume'] = False

    assert os.path.isfile(opt["train_data"] + '/labels.csv'), opt["train_data"] + "/labels.csv" + " does not exist"
    assert os.path.isfile(opt["valid_data"] + '/labels.csv'), opt["valid_data"] + "/labels.csv" + " does not exist"
    assert os.path.isfile(opt["test_data"] + '/labels.csv'), opt["test_data"] + "/labels.csv" + " does not exist"
    assert opt['experiment_name'] == get_fname(config_path), 'Experiment name in config file does not match with the file name'


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='config_files/en_filtered_config.yaml', help='Path to config file')
    args = parser.parse_args()

    opt = get_config(args.config)

    # safety_checks(opt, args.config)

    # os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    wandb.login()

    cudnn.benchmark = False
    cudnn.deterministic = False

    train(opt, amp=True, wandb_kwargs=opt['wandb_kwargs'])

