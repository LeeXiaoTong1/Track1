import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse
import os
import json
import shutil
import numpy as np
import random
import torchaudio
from sklearn.metrics import f1_score

from model import *
from dataset import *
from CSAM import *
from torch.utils.data import ConcatDataset, DataLoader, WeightedRandomSampler, Sampler
import torch.utils.data.sampler as torch_sampler
from backbone.rawaasist import *
from collections import defaultdict
from tqdm import tqdm, trange
from exp.feature_extraction_exp import *
from utils import *
import eval_metrics as em
from feature_extraction import *
import config

torch.set_default_tensor_type(torch.FloatTensor)
torch.multiprocessing.set_start_method('spawn', force=True)


def initParams():
    parser = config.initParams()

    # Training hyperparameters
    parser.add_argument('--num_epochs', type=int, default=20, help="Number of epochs for training")
    parser.add_argument('--batch_size', type=int, default=64, help="Mini batch size for training")
    parser.add_argument('--lr', type=float, default=0.0005, help="learning rate")
    parser.add_argument('--lr_decay', type=float, default=0.5, help="decay learning rate")
    parser.add_argument('--interval', type=int, default=4, help="interval to decay lr")
    parser.add_argument('--beta_1', type=float, default=0.9, help="bata_1 for Adam")
    parser.add_argument('--beta_2', type=float, default=0.999, help="beta_2 for Adam")
    parser.add_argument('--eps', type=float, default=1e-8, help="epsilon for Adam")
    parser.add_argument("--gpu", type=str, help="GPU index", default="7")
    parser.add_argument('--num_workers', type=int, default=8, help="number of workers")

    parser.add_argument('--train_task', type=str, default="atadd-track1",
                        choices=["atadd-track1", "atadd-track2"])
    parser.add_argument('--base_loss', type=str, default="ce", choices=["ce", "bce"],
                        help="use which loss for basic training")
    parser.add_argument('--continue_training', action='store_true',
                        help="continue training with trained model")

    parser.add_argument(
        '--save_best_by',
        type=str,
        default='loss',
        choices=['loss', 'eer', 'f1'],
        help='Metric used to save the best model: loss, eer, or f1'
    )

    # generalized strategy
    parser.add_argument('--SAM', type=bool, default=False, help="use SAM")
    parser.add_argument('--ASAM', type=bool, default=False, help="use ASAM")
    parser.add_argument('--CSAM', type=bool, default=False, help="use CSAM")

    ### PI-XLSR-AASIST
    parser.add_argument(
        "--pi_training",
        action="store_true",
        help="Use perturbation-invariant dual-view training"
    )

    parser.add_argument(
        "--pi_js",
        type=float,
        default=0.2,
        help="Weight of JS consistency loss"
    )

    parser.add_argument(
        "--pi_feat",
        type=float,
        default=0.05,
        help="Weight of feature consistency loss"
    )

    parser.add_argument(
        "--pi_aug_prob",
        type=float,
        default=1.0,
        help="Probability of applying perturbation to a batch"
    )
    ### PI-XLSR-AASIST
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    # Set seeds
    setup_seed(args.seed)

    if args.continue_training:
        pass
    else:
        if not os.path.exists(args.out_fold):
            os.makedirs(args.out_fold)
        else:
            shutil.rmtree(args.out_fold)
            os.mkdir(args.out_fold)

        if not os.path.exists(os.path.join(args.out_fold, 'checkpoint')):
            os.makedirs(os.path.join(args.out_fold, 'checkpoint'))
        else:
            shutil.rmtree(os.path.join(args.out_fold, 'checkpoint'))
            os.mkdir(os.path.join(args.out_fold, 'checkpoint'))

        with open(os.path.join(args.out_fold, 'args.json'), 'w') as file:
            json.dump(vars(args), file, indent=4)

        with open(os.path.join(args.out_fold, 'train_loss.log'), 'w') as file:
            file.write("epoch\tstep\ttrain_loss\n")

        with open(os.path.join(args.out_fold, 'dev_loss.log'), 'w') as file:
            file.write("epoch\tval_loss\tval_eer\tval_f1\n")

    args.cuda = torch.cuda.is_available()
    print('Cuda device available: ', args.cuda)
    args.device = torch.device("cuda" if args.cuda else "cpu")

    return args


def adjust_learning_rate(args, lr, optimizer, epoch_num):
    lr = lr * (args.lr_decay ** (epoch_num // args.interval))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def shuffle(feat, labels):
    shuffle_index = torch.randperm(labels.shape[0])
    feat = feat[shuffle_index]
    labels = labels[shuffle_index]
    return feat, labels
    
### PI-XLSR-AASIST
def fix_audio_length(x, target_len):
    cur_len = x.size(-1)
    if cur_len == target_len:
        return x
    
    if cur_len > target_len:
        start = random.randint(0, cur_len - target_len)
        return x[..., start:start + target_len]
    
    pad_len = target_len - cur_len
    return F.pad(x, (0, pad_len))

def random_perturb_batch(x, sample_rate=16000, aug_prob=1.0):
    """
    x: [B, T]
    return: [B, T]
    只做 signal-level perturbation，不引入外部音频数据。
    """
    if random.random() > aug_prob:
        return x

    target_len = x.size(-1)
    y = x.clone()

    op = random.choice([
        "resample",
        "speed",
        "noise",
        "gain",
        "clip"])

    if op == "resample":
        new_sr = random.choice([8000, 12000, 24000])
        y = torchaudio.functional.resample(y, sample_rate, new_sr)
        y = torchaudio.functional.resample(y, new_sr, sample_rate)
        y = fix_audio_length(y, target_len)

    elif op == "speed":
        factor = random.choice([0.90, 0.95, 1.05, 1.10])
        new_sr = int(sample_rate * factor)
        y = torchaudio.functional.resample(y, sample_rate, new_sr)
        y = fix_audio_length(y, target_len)

    elif op == "noise":
        snr_db = random.uniform(10, 30)
        signal_power = y.pow(2).mean(dim=-1, keepdim=True).clamp(min=1e-8)
        noise_power = signal_power / (10 ** (snr_db / 10))
        noise = torch.randn_like(y) * noise_power.sqrt()
        y = y + noise

    elif op == "gain":
        gain = random.uniform(0.5, 1.5)
        y = y * gain

    elif op == "clip":
        clip_val = random.uniform(0.6, 0.95)
        y = torch.clamp(y, -clip_val, clip_val)

    y = torch.clamp(y, -1.0, 1.0)
    return y

def js_divergence_from_logits(logits_1, logits_2):
    p = F.softmax(logits_1, dim=1)
    q = F.softmax(logits_2, dim=1)
    m = 0.5 * (p + q)
    js = 0.5 * (F.kl_div(torch.log(p + 1e-8), m, reduction="batchmean") + F.kl_div(torch.log(q + 1e-8), m, reduction="batchmean"))
    return js
### PI-XLSR-AASIST

def train(args):
    torch.set_default_tensor_type(torch.FloatTensor)

    # initialize model
    # Conventional CM 
    if args.model == 'aasist':
        feat_model = Rawaasist().to(args.device)
    if args.model == 'specresnet':
        feat_model = ResNet18ForAudio().to(args.device)
    #❄️ FR SSL-based CM
    if args.model == 'fr-w2v2aasist':
        feat_model = XLSRAASIST(model_dir=args.xlsr).to(args.device)
    if args.model == 'fr-wavlmaasist':
        feat_model = WAVLMAASIST(model_dir=args.wavlm).to(args.device)
    if args.model == 'fr-mertaasist':
        feat_model = MERTAASIST(model_dir=args.mert).to(args.device)
    #🔥 FT-SSL-based CM
    if args.model == 'ft-w2v2aasist':
        feat_model = XLSRAASIST(model_dir=args.xlsr, freeze=False).to(args.device)
    if args.model == 'ft-wavlmaasist':
        feat_model = WAVLMAASIST(model_dir=args.wavlm, freeze=False).to(args.device)
    if args.model == 'ft-mertaasist':
        feat_model = MERTAASIST(model_dir=args.mert, freeze=False).to(args.device)
    #🔥 WPT-SSL-based CM 
    if args.model == 'pt-w2v2aasist':
        feat_model = PTW2V2AASIST(
            model_dir=args.xlsr,
            prompt_dim=args.prompt_dim,
            num_prompt_tokens=args.num_prompt_tokens,
            dropout=args.pt_dropout
        ).to(args.device)
    if args.model == "wpt-w2v2aasist":
        feat_model = WPTW2V2AASIST(
            model_dir=args.xlsr,
            prompt_dim=args.prompt_dim,
            num_prompt_tokens=args.num_prompt_tokens,
            num_wavelet_tokens=args.num_wavelet_tokens,
            dropout=args.pt_dropout
        ).to(args.device)
    # 修改 aalf
    if args.model == 'aalf-w2v2aasist':
        layer_ids = tuple(int(x) for x in args.aalf_layers.split(","))

        feat_model = AALFXLSRAASIST(
            model_dir=args.xlsr,
            freeze=args.aalf_freeze_xlsr,
            layer_ids=layer_ids,
            bottleneck=args.aalf_bottleneck,
            topk=args.aalf_topk,
            dropout=args.aalf_dropout
        ).to(args.device)
    
    feat_optimizer = torch.optim.Adam(
        feat_model.parameters(),
        lr=args.lr,
        betas=(args.beta_1, args.beta_2),
        eps=args.eps,
        weight_decay=0.0005
    )
    # 修改
    
    if getattr(args, "init_from", ""):
        print(f"Loading initialization checkpoint from: {args.init_from}")
        ckpt = torch.load(args.init_from, map_location=args.device)
        missing, unexpected = feat_model.load_state_dict(ckpt, strict=False)
        print("Missing keys:", missing)
        print("Unexpected keys:", unexpected)

    if args.SAM or args.CSAM:
        base_optimizer = torch.optim.Adam
        feat_optimizer = SAM(
            feat_model.parameters(),
            base_optimizer,
            lr=args.lr,
            betas=(args.beta_1, args.beta_2),
            weight_decay=0.0005
        )

    if args.train_task == "atadd-track1":
        atadd_t1_trainset = atadd_dataset(
            args.atadd_t1_train_audio,
            args.atadd_t1_train_label,
            audio_length=args.audio_len
        )
        atadd_t1_devset = atadd_dataset(
            args.atadd_t1_dev_audio,
            args.atadd_t1_dev_label,
            audio_length=args.audio_len
        )
        train_set = [atadd_t1_trainset]
        dev_set = [atadd_t1_devset]

    if args.train_task == "atadd-track2":
        atadd_t2_trainset = atadd_dataset(
            args.atadd_t2_train_audio,
            args.atadd_t2_train_label,
            audio_length=args.audio_len
        )
        atadd_t2_devset = atadd_dataset(
            args.atadd_t2_dev_audio,
            args.atadd_t2_dev_label,
            audio_length=args.audio_len
        )
        train_set = [atadd_t2_trainset]
        dev_set = [atadd_t2_devset]

    for dataset in train_set:
        print(len(dataset), f"Dataset {dataset} length")
        assert len(dataset) > 0, f"Dataset {dataset} is empty. Please check the dataset loading process."
    for dataset in dev_set:
        print(len(dataset), f"Dataset {dataset} length")
        assert len(dataset) > 0, f"Dataset {dataset} is empty. Please check the dataset loading process."

    training_set = ConcatDataset(train_set)
    validation_set = ConcatDataset(dev_set)

    trainOriDataLoader = DataLoader(
        training_set,
        batch_size=int(args.batch_size),
        shuffle=False,
        num_workers=args.num_workers,
        sampler=torch_sampler.SubsetRandomSampler(range(len(training_set))),
        pin_memory=args.cuda
    )

    valOriDataLoader = DataLoader(
        validation_set,
        batch_size=int(args.batch_size),
        shuffle=False,
        num_workers=args.num_workers,
        sampler=torch_sampler.SubsetRandomSampler(range(len(validation_set))),
        pin_memory=args.cuda
    )

    trainOri_flow = iter(trainOriDataLoader)
    valOri_flow = iter(valOriDataLoader)

    if args.train_task == "atadd-track1":
        weight = torch.FloatTensor([4, 1]).to(args.device)

    if args.train_task == "atadd-track2":
        weight = torch.FloatTensor([3.5, 1]).to(args.device)

    print(f"Using class weight: {weight.tolist()}")
    print(f"Best model will be saved by: {args.save_best_by}")

    if args.base_loss == "ce":
        criterion = nn.CrossEntropyLoss(weight=weight)
    else:
        criterion = nn.BCEWithLogitsLoss()

    prev_loss = float("inf")
    prev_eer = float("inf")
    prev_f1 = -float("inf")
    monitor_loss = 'base_loss'

    for epoch_num in tqdm(range(args.num_epochs)):
        feat_model.train()
        trainlossDict = defaultdict(list)
        devlossDict = defaultdict(list)

        adjust_learning_rate(args, args.lr, feat_optimizer, epoch_num)

        for i in trange(0, len(trainOriDataLoader), total=len(trainOriDataLoader), initial=0):
            try:
                feat, audio_fn, labels = next(trainOri_flow)
            except StopIteration:
                trainOri_flow = iter(trainOriDataLoader)
                feat, audio_fn, labels = next(trainOri_flow)

            feat = feat.to(args.device, non_blocking=True)
            labels = labels.to(args.device, non_blocking=True)

            if args.SAM or args.ASAM or args.CSAM:
                enable_running_stats(feat_model)
                feats, feat_outputs = feat_model(feat)
                feat_loss = criterion(feat_outputs, labels)
                feat_loss.mean().backward()
                feat_optimizer.first_step(zero_grad=True)

                disable_running_stats(feat_model)
                feats, feat_outputs = feat_model(feat)
                criterion(feat_outputs, labels).mean().backward()
                feat_optimizer.second_step(zero_grad=True)

            # else:
            #     feat_optimizer.zero_grad()
            #     feats, feat_outputs = feat_model(feat)
            #     feat_loss = criterion(feat_outputs, labels)
            #     feat_loss.backward()
            #     feat_optimizer.step()
            
            ### PI-XLSR-AASIST
            else:
                feat_optimizer.zero_grad()

                if args.pi_training:
                    feat_aug = random_perturb_batch(
                        feat,
                        sample_rate=16000,
                        aug_prob=args.pi_aug_prob
                    )

                    feats_clean, outputs_clean = feat_model(feat)
                    feats_aug, outputs_aug = feat_model(feat_aug)

                    ce_clean = criterion(outputs_clean, labels)
                    ce_aug = criterion(outputs_aug, labels)

                    js_loss = js_divergence_from_logits(outputs_clean, outputs_aug)

                    feat_consistency = F.mse_loss(
                        F.normalize(feats_clean, dim=-1),
                        F.normalize(feats_aug, dim=-1)
                    )

                    feat_loss = (
                        0.5 * ce_clean +
                        0.5 * ce_aug +
                        args.pi_js * js_loss +
                        args.pi_feat * feat_consistency
                    )

                else:
                    feats, feat_outputs = feat_model(feat)
                    feat_loss = criterion(feat_outputs, labels)

                feat_loss.backward()
                feat_optimizer.step()
            ### PI-XLSR-AASIST

            trainlossDict['base_loss'].append(feat_loss.item())

            with open(os.path.join(args.out_fold, "train_loss.log"), "a") as log:
                log.write(
                    str(epoch_num) + "\t" +
                    str(i) + "\t" +
                    str(trainlossDict[monitor_loss][-1]) + "\n"
                )

        feat_model.eval()
        with torch.no_grad():
            ip1_loader, tag_loader, idx_loader, score_loader, pred_loader = [], [], [], [], []

            for i in trange(0, len(valOriDataLoader), total=len(valOriDataLoader), initial=0):
                try:
                    feat, audio_fn, labels = next(valOri_flow)
                except StopIteration:
                    valOri_flow = iter(valOriDataLoader)
                    feat, audio_fn, labels = next(valOri_flow)

                feat = feat.to(args.device, non_blocking=True)
                labels = labels.to(args.device, non_blocking=True)

                feats, feat_outputs = feat_model(feat)

                if args.base_loss == "bce":
                    feat_loss = criterion(feat_outputs, labels.unsqueeze(1).float())
                    score = torch.sigmoid(feat_outputs[:, 0])
                    pred = torch.where(score >= 0.5,
                                       torch.zeros_like(labels),
                                       torch.ones_like(labels))
                else:
                    feat_loss = criterion(feat_outputs, labels)
                    prob = F.softmax(feat_outputs, dim=1)
                    score = prob[:, 0]   
                    pred = torch.where(score >= 0.5,
                                       torch.zeros_like(labels),
                                       torch.ones_like(labels))

                ip1_loader.append(feats)
                idx_loader.append(labels)
                pred_loader.append(pred)
                devlossDict["base_loss"].append(feat_loss.item())
                score_loader.append(score)

                desc_str = ''
                for key in sorted(devlossDict.keys()):
                    desc_str += key + ':%.5f' % (np.nanmean(devlossDict[key])) + ', '


            valLoss = np.nanmean(devlossDict[monitor_loss])
            scores = torch.cat(score_loader, 0).data.cpu().numpy()
            labels = torch.cat(idx_loader, 0).data.cpu().numpy()
            preds = torch.cat(pred_loader, 0).data.cpu().numpy()

            val_eer = em.compute_eer(scores[labels == 0], scores[labels == 1])[0]
            val_f1 = f1_score(labels, preds, average='macro')

            with open(os.path.join(args.out_fold, "dev_loss.log"), "a") as log:
                log.write(
                    str(epoch_num) + "\t" +
                    str(valLoss) + "\t" +
                    str(val_eer) + "\t" +
                    str(val_f1) + "\n"
                )

            print("Val Loss: {}".format(valLoss))
            print("Val EER: {}".format(val_eer))
            print("Val F1 : {}".format(val_f1))

        if (epoch_num + 1) % 5 == 0:
            torch.save(
                feat_model.state_dict(),
                os.path.join(args.out_fold, 'checkpoint', 'atadd_model_%d.pt' % (epoch_num + 1))
            )

        save_flag = False

        if args.save_best_by == "loss":
            if valLoss < prev_loss:
                prev_loss = valLoss
                save_flag = True

        elif args.save_best_by == "eer":
            if val_eer < prev_eer:
                prev_eer = val_eer
                save_flag = True

        elif args.save_best_by == "f1":
            if val_f1 > prev_f1:
                prev_f1 = val_f1
                save_flag = True

        if save_flag:
            torch.save(
                feat_model.state_dict(),
                os.path.join(args.out_fold, 'atadd_model.pt')
            )
            print(f"Best model updated by {args.save_best_by} at epoch {epoch_num}")

    return feat_model


if __name__ == "__main__":
    args = initParams()
    train(args)
