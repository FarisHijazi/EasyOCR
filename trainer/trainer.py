import os
import wandb

os.environ['WANDB_API_KEY'] = "***REMOVED***"
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
    else:
        opt.character = opt.number + opt.symbol + opt.lang_char
    os.makedirs(f'./saved_models/{opt.experiment_name}', exist_ok=True)
    return opt


def safety_checks(opt):
    try:
        opt['saved_model'] = sorted(glob.glob(opt['saved_model']), key=lambda t: os.stat(t).st_mtime)[-1]
        print('saved_model', opt['saved_model'])
    except Exception as e:
        print('failed to find saved_model', e, 'falling back to base_model: ', end='')
        opt['saved_model'] = opt['base_model']
        print(opt['base_model'])
        assert os.path.isfile(opt['base_model'])

    assert os.path.isfile(opt["train_data"] + '/labels.csv'), opt["train_data"] + "/labels.csv" + " does not exist"
    assert os.path.isfile(opt["valid_data"] + '/labels.csv'), opt["valid_data"] + "/labels.csv" + " does not exist"
    assert os.path.isfile(opt["test_data"] + '/labels.csv'), opt["test_data"] + "/labels.csv" + " does not exist"


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='config_files/en_filtered_config.yaml', help='Path to config file')
    args = parser.parse_args()

    opt = get_config(args.config)

    safety_checks(opt)

    # os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    wandb.login()

    cudnn.benchmark = False
    cudnn.deterministic = False

    train(opt, amp=True)

