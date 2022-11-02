"""
this script converts a pretrained model to a model that can be used by the trainer
without this script, you won't be able to load the pretrained models to finetune them, this script will modify the size of the output layer to match the number of characters in the dataset

note that most pretrained models (at least the arabic g1) have the following parameters:

input_channel: 1
output_channel: 512 #256
hidden_size: 512 #256

be sure to modify them in your config file
"""

import argparse
import torch


if __name__ == "__main__":
    parser = argparse.ArgumentParser(__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('pretrained_model_path', type=str, help='Path to pretrained model')
    parser.add_argument('n_vocab', type=int, help='Number of vocab+1 (count all the symbols and lang_char and numbers and add +1)')
    parser.add_argument('--method', type=str, default='reinit', choices={'reinit', 'truncate'},
                        help='Method to use to convert the model, reinit: reinitialize weights.'
                            'truncate: just remove the last rows of the matrix to match the size (this method won\'t work if the new vocab is bigger than the old one')
    args = parser.parse_args()

    pretrained_model_path = args.pretrained_model_path
    pretrained_state_dict = torch.load(pretrained_model_path, map_location='cpu')


    if args.method == 'trunctate':
        pretrained_state_dict["module.Prediction.weight"] = pretrained_state_dict["module.Prediction.weight"][:args.n_vocab]
        pretrained_state_dict["module.Prediction.bias"] = pretrained_state_dict["module.Prediction.bias"][:args.n_vocab]
    else:
        pretrained_state_dict["module.Prediction.weight"] = torch.randn((args.n_vocab, pretrained_state_dict["module.Prediction.weight"].shape[1]), requires_grad=True)
        pretrained_state_dict["module.Prediction.bias"] =  torch.zeros((args.n_vocab), requires_grad=True)

    outfpath = pretrained_model_path.replace('.pth', f'_vocab={args.n_vocab}.pth')
    print('saving to', outfpath)
    torch.save(pretrained_state_dict, outfpath)
