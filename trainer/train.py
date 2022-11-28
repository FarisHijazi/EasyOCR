import os
import sys
import time
import random
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.init as init
import torch.optim as optim
import torch.utils.data
from torch.cuda.amp import autocast, GradScaler
import numpy as np
import wandb
from tqdm.auto import tqdm
from copy import deepcopy

from utils import CTCLabelConverter, AttnLabelConverter, Averager
from dataset import hierarchical_dataset, AlignCollate, Batch_Balanced_Dataset
from model import Model
from test import validation

from utils import AttrDict
import warnings
import jiwer

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
ignore_hparams = [
    'experiment_name',
    'workers',
    'num_iter',
    'valInterval',
    'displayInterval',
    'saveInterval',
    'wandb_kwargs',
]

def asciify_dict_recursive(d):
    if isinstance(d, dict):
        return {asciify_dict_recursive(k): asciify_dict_recursive(v) for k, v in d.items()}
    elif isinstance(d, list):
        return [asciify_dict_recursive(v) for v in d]
    elif isinstance(d, str):
        return d.encode('unicode-escape', 'ignore').decode('ascii')
    else:
        return d

def count_parameters(model):
    print("Modules, Parameters")
    total_params = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad: continue
        param = parameter.numel()
        #table.add_row([name, param])
        total_params+=param
        print(name, param)
    print(f"Total Trainable Params: {total_params}")
    return total_params

def train(opt, show_number = 2, amp=False, wandb_kwargs={}):
    """ dataset preparation """
    if not opt.data_filtering_off:
        print('Filtering the images containing characters which are not in opt.character')
        print('Filtering the images whose label is longer than opt.batch_max_length')


    opt.test_data = opt.get('test_data', 'valid_data')
    assert os.path.exists(opt["train_data"]), opt["train_data"] + " does not exist"
    assert os.path.exists(opt["valid_data"]), opt["valid_data"] + " does not exist"
    assert os.path.exists(opt["test_data"]), opt["test_data"] + " does not exist"

    opt.select_data = opt.select_data.split('-')
    opt.batch_ratio = opt.batch_ratio.split('-')
    train_dataset = Batch_Balanced_Dataset(opt)

    log = open(f'./saved_models/{opt.experiment_name}/log_dataset.txt', 'a', encoding="utf8")
    AlignCollate_valid = AlignCollate(imgH=opt.imgH, imgW=opt.imgW, keep_ratio_with_pad=opt.PAD, contrast_adjust=opt.contrast_adjust)
    valid_dataset, valid_dataset_log = hierarchical_dataset(root=opt.valid_data, opt=opt)
    valid_loader = torch.utils.data.DataLoader(
        valid_dataset, batch_size=min(32, opt.batch_size),
        shuffle=True,  # 'True' to check training progress with validation function.
        num_workers=int(opt.workers), prefetch_factor=512,
        collate_fn=AlignCollate_valid, pin_memory=True)

    if opt.get('test_data', 'valid_data') == opt['valid_data']:
        test_loader = iter(valid_loader)
    else:
        opt_test = AttrDict(deepcopy(opt))
        opt_test['data_filtering_off'] = True
        opt_test['batch_ratio'] = 1
        opt_test['valid_data'] = opt_test['test_data'] 
        test_dataset, test_dataset_log = hierarchical_dataset(root=opt.test_data, opt=opt_test)
        print('test_dataset', len(test_dataset))
        test_loader = torch.utils.data.DataLoader(
            test_dataset, batch_size=min(32, opt.batch_size),
            shuffle=True,  # 'True' to check training progress with validation function.
            num_workers=int(opt.workers), prefetch_factor=512,
            collate_fn=AlignCollate_valid, pin_memory=True)
    log.write(valid_dataset_log)
    print('-' * 80)
    log.write('-' * 80 + '\n')
    log.close()
    
    """ model configuration """
    if 'CTC' in opt.Prediction:
        converter = CTCLabelConverter(opt.character)
    else:
        converter = AttnLabelConverter(opt.character)
    opt.num_class = len(converter.character)

    if opt.rgb:
        opt.input_channel = 3
    model = Model(opt)
    print('model input parameters', opt.imgH, opt.imgW, opt.num_fiducial, opt.input_channel, opt.output_channel,
          opt.hidden_size, opt.num_class, opt.batch_max_length, opt.Transformation, opt.FeatureExtraction,
          opt.SequenceModeling, opt.Prediction)

    if opt.saved_model != '':
        pretrained_dict = torch.load(opt.saved_model)
        if opt.new_prediction:
            model.Prediction = nn.Linear(model.SequenceModeling_output, len(pretrained_dict['module.Prediction.weight']))  
        
        model = torch.nn.DataParallel(model).to(device) 
        print(f'loading pretrained model from {opt.saved_model}')
        if opt.FT:
            model.load_state_dict(pretrained_dict, strict=False)
        else:
            model.load_state_dict(pretrained_dict)
        if opt.new_prediction:
            model.module.Prediction = nn.Linear(model.module.SequenceModeling_output, opt.num_class)  
            for name, param in model.module.Prediction.named_parameters():
                if 'bias' in name:
                    init.constant_(param, 0.0)
                elif 'weight' in name:
                    init.kaiming_normal_(param)
            model = model.to(device) 
    else:
        # weight initialization
        for name, param in model.named_parameters():
            if 'localization_fc2' in name:
                print(f'Skip {name} as it is already initialized')
                continue
            try:
                if 'bias' in name:
                    init.constant_(param, 0.0)
                elif 'weight' in name:
                    init.kaiming_normal_(param)
            except Exception as e:  # for batchnorm.
                if 'weight' in name:
                    param.data.fill_(1)
                continue
        model = torch.nn.DataParallel(model).to(device)
    
    model.train() 
    print("Model:")
    print(model)
    count_parameters(model)
    
    """ setup loss """
    if 'CTC' in opt.Prediction:
        criterion = torch.nn.CTCLoss(zero_infinity=True).to(device)
    else:
        criterion = torch.nn.CrossEntropyLoss(ignore_index=0).to(device)  # ignore [GO] token = ignore index 0
    # loss averager
    loss_avg = Averager()

    # freeze some layers
    try:
        if opt.freeze_FeatureFxtraction:
            for param in model.module.FeatureExtraction.parameters():
                param.requires_grad = False
        if opt.freeze_SequenceModeling:
            for param in model.module.SequenceModeling.parameters():
                param.requires_grad = False
    except:
        pass
    
    # filter that only require gradient decent
    filtered_parameters = []
    params_num = []
    for p in filter(lambda p: p.requires_grad, model.parameters()):
        filtered_parameters.append(p)
        params_num.append(np.prod(p.size()))
    print('Trainable params num : ', sum(params_num))
    # [print(name, p.numel()) for name, p in filter(lambda p: p[1].requires_grad, model.named_parameters())]

    # setup optimizer
    if opt.optim=='adam':
        #optimizer = optim.Adam(filtered_parameters, lr=opt.lr, betas=(opt.beta1, 0.999))
        optimizer = optim.Adam(filtered_parameters)
    else:
        optimizer = optim.Adadelta(filtered_parameters, lr=opt.lr, rho=opt.rho, eps=opt.eps)
    print("Optimizer:")
    print(optimizer)

    """ final options """
    # print(opt)
    with open(f'./saved_models/{opt.experiment_name}/opt.txt', 'a', encoding="utf8") as opt_file:
        opt_log = '------------ Options -------------\n'
        args = vars(opt)
        for k, v in args.items():
            opt_log += f'{str(k)}: {str(v)}\n'
        opt_log += '---------------------------------------\n'
        print(opt_log)
        opt_file.write(opt_log)

    """ start training """
    start_iter = 0
    if opt.saved_model != '':
        try:
            start_iter = int(opt.saved_model.split('_')[-1].split('.')[0])
            print(f'continue to train, start_iter: {start_iter}')
        except:
            pass

    wandb.init(
        project=opt.get("project_name", os.environ.get("WANDB_PROJECT", None)),
        name=opt['experiment_name'],
        config={k: v for k, v in asciify_dict_recursive(deepcopy(opt)).items() if k not in ignore_hparams},
        **wandb_kwargs
    )

    # # EXPERIMENTAL!!! set global step to start_iter
    print(f'setting wandb.run._starting_step from {wandb.run._starting_step} to {start_iter}')
    wandb.run._starting_step = start_iter
    print(f'setting wandb.run._step from {wandb.run._step} to {start_iter}')
    wandb.run._step = start_iter


    start_time = time.time()
    best_accuracy = -1
    best_norm_ED = -1
    i = start_iter

    scaler = GradScaler(enabled=amp)
    t1= time.time()
    step_start_time = time.time()
    pbar = tqdm(train_dataset, 'Training', total=opt['num_iter'], initial=start_iter, unit='batch')
    for (image_tensors, labels) in pbar:
        # print(f'{time.time():.2f}s: loaded data')

        data_duration = time.time() - step_start_time
        # train part
        optimizer.zero_grad(set_to_none=True)
    
        # print(f'{time.time():.2f}s: starting model training step')
        with autocast(enabled=amp):
            image = image_tensors.to(device)
            text, length = converter.encode(labels, batch_max_length=opt.batch_max_length)
            batch_size = image.size(0)

            if 'CTC' in opt.Prediction:
                preds = model(image, text).log_softmax(2)
                preds_size = torch.IntTensor([preds.size(1)] * batch_size)
                preds = preds.permute(1, 0, 2)
                torch.backends.cudnn.enabled = False
                cost = criterion(preds, text.to(device), preds_size.to(device), length.to(device))
                torch.backends.cudnn.enabled = True
            else:
                preds = model(image, text[:, :-1])  # align with Attention.forward
                target = text[:, 1:]  # without [GO] Symbol
                cost = criterion(preds.view(-1, preds.shape[-1]), target.contiguous().view(-1))
            scaler.scale(cost).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), opt.grad_clip)
            scaler.step(optimizer)
            scaler.update()
        # print(f'{time.time():.2f}s: ending model training step')

        loss_avg.add(cost)
        model_duration = time.time() - step_start_time

        # log to wandb
        if i % opt.get('displayInterval', 10) == 0:
            wandb.log({"Train Loss": loss_avg.val()}, step=i)
            # loss_avg.reset()
        # validation part
        isValInterval = (i % opt.valInterval == 0) # yes I'm running eval on step 0 because I want to see the initial accuracy
        if isValInterval:
            print("Running EVALUATION. training time: ", time.time() - t1)
            t1=time.time()
            elapsed_time = time.time() - start_time
            # for log
            with open(f'./saved_models/{opt.experiment_name}/log_train.txt', 'a', encoding="utf8") as log:
                
                with autocast(enabled=amp):
                    model.eval()
                    with torch.no_grad():
                        valid_loss, current_accuracy, current_norm_ED, preds, confidence_score, labels,\
                        infer_time, length_of_data = validation(model, criterion, valid_loader, converter, opt, device)

                    if opt.get('test_data', 'valid_data') == opt['valid_data']:
                        test_loss, test_current_accuracy, test_current_norm_ED, test_preds, test_confidence_score, test_labels,\
                        test_infer_time, test_length_of_data = valid_loss, current_accuracy, current_norm_ED, preds, confidence_score, labels,\
                        infer_time, length_of_data
                    else:
                        with torch.no_grad():
                            test_loss, test_current_accuracy, test_current_norm_ED, test_preds, test_confidence_score, test_labels,\
                            test_infer_time, test_length_of_data = validation(model, criterion, test_loader, converter, opt, device)
                    model.train()

                # training loss and validation loss
                loss_log = f'[{i}/{opt.num_iter}] Train loss: {loss_avg.val():0.5f}, Valid loss: {valid_loss:0.5f}, Elapsed_time: {elapsed_time:0.5f}'
                loss_avg.reset()

                current_model_log = f'{"Current_accuracy":17s}: {current_accuracy:0.3f}, {"Current_norm_ED":17s}: {current_norm_ED:0.4f}'

                # keep best accuracy model (on valid dataset)
                if current_accuracy > best_accuracy:
                    best_accuracy = current_accuracy
                    torch.save(model.state_dict(), f'./saved_models/{opt.experiment_name}/best_accuracy.pth')
                    
                    try:
                        #TODO: learn to save artifacts
                        # wandb.save(f'./saved_models/{opt.experiment_name}/best_accuracy.pth')
                        pass
                    except Exception as e:
                        warnings.warn(f'wandb save failed: {e}')

                if current_norm_ED > best_norm_ED:
                    best_norm_ED = current_norm_ED
                    torch.save(model.state_dict(), f'./saved_models/{opt.experiment_name}/best_norm_ED.pth')
                    
                    try:
                        # wandb.save(f'./saved_models/{opt.experiment_name}/best_norm_ED.pth')
                        pass
                    except Exception as e:
                        warnings.warn(f'wandb save failed: {e}')


                best_model_log = f'{"Best_accuracy":17s}: {best_accuracy:0.3f}, {"Best_norm_ED":17s}: {best_norm_ED:0.4f}'

                loss_model_log = f'{loss_log}\n{current_model_log}\n{best_model_log}'
                # print(loss_model_log)
                log.write(loss_model_log + '\n')
                wandb.log({
                    "Valid Loss": valid_loss,
                    "Valid Accuracy": current_accuracy,
                    "Valid CER": jiwer.cer(labels, preds),
                    "Valid Norm_ED": current_norm_ED,

                    "Test Loss": test_loss,
                    "Test Accuracy": test_current_accuracy,
                    "Test Norm_ED": test_current_norm_ED,
                    "Test CER": jiwer.cer(test_labels, test_preds),
                }, step=i)

                # show some predicted results
                dashed_line = '-' * 80
                head = f'{"Ground Truth":25s} | {"Prediction":25s} | Confidence Score & T/F'
                predicted_result_log = f'{dashed_line}\n{head}\n{dashed_line}\n'
                
                #show_number = min(show_number, len(labels))
                predicted_result_table = []
                start = random.randint(0,len(labels) - show_number )    
                for gt, pred, confidence in zip(labels[start:start+show_number], preds[start:start+show_number], confidence_score[start:start+show_number]):
                    if 'Attn' in opt.Prediction:
                        gt = gt[:gt.find('[s]')]
                        pred = pred[:pred.find('[s]')]
                        predicted_result_table.append([gt, pred, f'{confidence:0.4f}\t{str(pred == gt)}'])

                    predicted_result_log += f'{gt:25s} | {pred:25s} | {confidence:0.4f}\t{str(pred == gt)}\n'
                predicted_result_log += f'{dashed_line}'
                log.write(predicted_result_log + '\n')
                # wandb.log({"Predicted Result": wandb.Html(predicted_result_log.encode('ascii', 'xmlcharrefreplace').decode('ascii'))}, step=i)
                table = wandb.Table(columns=['Ground Truth', "Prediction", "Confidence Score & T/F"], data=predicted_result_table)
                wandb.log({"Validation Result Table": table}, step=i)
                print("Validation time: ", time.time()-t1)
                t1=time.time()

        # save model per 1e+4 iter.
        if (i + 1) % opt.get('saveInterval', 10000) == 0:
            save_path = f'./saved_models/{opt.experiment_name}/iter_{i+1}.pth'
            torch.save(model.state_dict(), save_path)
            
            try:
                # wandb.save(save_path)
                pass
            except Exception as e:
                warnings.warn(f'wandb save failed: {e}')



        if i == opt.num_iter:
            print('end the training')
            wandb.finish()
            sys.exit()

        step_duration = time.time() - step_start_time
        if data_duration / step_duration * 100 > 30 and not isValInterval:
            warnings.warn(f'Warning: Your dataloader is too slow ({data_duration} seconds/iteration).')

        i += 1
        step_start_time = time.time()
        pbar.set_description(f'[{i}/{opt.num_iter}]')
        pbar.set_postfix({'TrainLoss': loss_avg.val(), 'Data duration': f'{data_duration / step_duration * 100:0.1f}%', "model duration": f"{model_duration / step_duration * 100:0.1f}%"})
        pbar.update(1)

        # print percentage of time spent on each part of the training loop
        # print(f"{'Data Loading':20s}: {data_duration / step_duration * 100:0.1f}%, {'Model Running':20s}: {model_duration / step_duration * 100:0.1f}%")
        # print(f'{time.time():.2f}s: beginning data load')

