print("这个代码使用的是aishell3 88w数据集的前100条做的训练")
#========== wav2mel.py
print('运行wav2mel.py')



import os
import glob
import tqdm
import torch
import argparse
import numpy as np
from mtts.utils.stft import TacotronSTFT
from scipy.io.wavfile import read
from mtts.utils.logging import get_logger
import librosa
import yaml

logger = get_logger(__file__)

# data: http://aishell-3.oss-cn-beijing.aliyuncs.com/AISHELL-3%20ReadMe.pdf
def read_wav_np(path):
    sr, wav = read(path)
    if len(wav.shape) == 2:
        wav = wav[:, 0]
    if wav.dtype == np.int16:
        wav = wav / 32768.0
    elif wav.dtype == np.int32:
        wav = wav / 2147483648.0
    elif wav.dtype == np.uint8:
        wav = (wav - 128) / 128.0
    wav = wav.astype(np.float32)
    return sr, wav


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, required=False,
                        help="yaml file for config.")
    parser.add_argument('-w', '--wav_path', type=str, required=False,
                        help="root directory of wav files")
    parser.add_argument('-m', '--mel_path', type=str, required=False,
                        help="root directory of mel files")
    parser.add_argument('-d', '--device', type=str, required=False,
                        help="device, cpu or cuda:0, cuda:1,...")
    parser.add_argument('-r', '--resample_mode', type=str, required=False, default='kaiser_fast',
                        help="use kaiser_best for high-quality audio")

    args = parser.parse_args()
    args.config='aishell3/config.yaml'
    args.wav_path='aishell3/outputs'
    args.mel_path='aishell3/outputs_mel'

    import os
    import sys
    import shutil

    # Get directory name
    shutil.rmtree(args.mel_path)
    args.device='cpu'


    logger.info(f'using resample mode {args.resample_mode}')
    with open(args.config) as f:
        config = yaml.safe_load(f)
    logger.info('loading TacotronSTFT')  # 引入快速傅里叶算法
    stft = TacotronSTFT(filter_length=config['fbank']['n_fft'],
                        hop_length=config['fbank']['hop_length'],
                        win_length=config['fbank']['win_length'],
                        n_mel_channels=config['fbank']['n_mels'],
                        sampling_rate=config['fbank']['sample_rate'],
                        mel_fmin=config['fbank']['fmin'],
                        mel_fmax=config['fbank']['fmax'],
                        device=args.device)

    logger.info('done')
    wav_files = glob.glob(os.path.join(args.wav_path, '*.wav'), recursive=False)
    logger.info(f'{len(wav_files)} found in {args.wav_path}')
    mel_path = args.mel_path
    logger.info(f'mel files will be saved to {mel_path}')

    # Create all folders
    os.makedirs(mel_path, exist_ok=True)
    for wavpath in tqdm.tqdm(wav_files, desc='preprocess wav to mel'):
        wav, r = librosa.load(wavpath, sr=config['fbank']['sample_rate'], res_type=args.resample_mode)
        wav = torch.from_numpy(wav).unsqueeze(0) # 音频转化为 tensor
        mel = stft.mel_spectrogram(wav)  # mel [1, num_mel, T] # 转化为mel序列

        mel = mel.squeeze(0)  # [num_mel, T]
        id = os.path.basename(wavpath).split(".")[0]
        np.save('{}/{}.npy'.format(mel_path, id), mel.numpy(), allow_pickle=False)











print('prepare the scp files')
import glob
import os
import argparse

from mtts.utils.logging import get_logger

logger = get_logger(__file__)


def augment_cn_with_sil(py_sent, cn_sent):
    sil_loc = [i for i, p in enumerate(py_sent.split()) if p == 'sil']
    han = [h for i, h in enumerate(cn_sent.split()) if h != 'sil']

    k = 0
    final = []
    for i in range(len(han) + len(sil_loc)):
        if i in sil_loc:
            final += ['sil']
        else:
            final += [han[k]]
            k += 1
    return ' '.join(final)


def write_scp(filename, scp):
    # shutil.rmtree(filename)
    with open(filename, 'wt') as f:
        f.write('\n'.join(scp) + '\n')


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='data pre-processing')
    parser.add_argument('--meta_file', type=str, required=False, default='name_py_hz_dur.txt')
    parser.add_argument('--wav_folder', type=str, required=False, default='./wavs')
    parser.add_argument('--mel_folder', type=str, required=False, default='./mels')
    parser.add_argument('--dst_folder', type=str, required=False, default='./train')
    parser.add_argument('--generate_vocab', type=bool, required=False, default=False)

    args = parser.parse_args()




    args.wav_folder='aishell3/outputs'
    args.meta_file='input.txt'
    args.mel_folder='aishell3/outputs_mel'
    args.device='cpu'






    logger.info(args)

    lines = open(args.meta_file).read().split('\n')
    lines = [l.split('|') for l in lines if len(l) > 0]
    files = glob.glob(f'{args.wav_folder}/*.wav')
    print('训练的音频文件数量')
    logger.info(f'{len(files)} wav files found')
    lines0 = []
    lines=[i for i in lines if i[0][0]!='#']
    for name, py, gp, dur in lines:
        gp = augment_cn_with_sil(py, gp)  # make sure gp and py has the same # of sil   py是拼音 gp是中文
        assert len(py.split()) == len(gp.split()), f'error in {name}:{py},{gp}'
        lines0 += [(name, py, gp, dur)]
    lines = lines0

    wav_scp = []
    mel_scp = []
    gp_scp = []
    py_scp = []
    dur_scp = []
    spk_scp = []
    all_spk = []
    all_spk = [l[0][:7] for l in lines] #对于当前小数据集的测试,认为每句话都是不同speaker说的.
    all_spk = list(set(all_spk))
    all_spk.sort()
    spk2idx = {s: i for i, s in enumerate(all_spk)}
    all_py = []
    all_gp = []

    for name, py, gp, dur in lines:
        wav_scp += [name + ' ' + f'{args.wav_folder}/{name}.wav']
        mel_scp += [name + ' ' + f'{args.mel_folder}/{name}.npy']
        py_scp += [name + ' ' + py]
        gp_scp += [name + ' ' + gp]
        dur_scp += [name + ' ' + dur]
        n = len(gp.split())
        spk_idx = spk2idx[name[:7]]
        spk_scp += [name + ' ' + ' '.join([str(spk_idx)] * n)]
    if args.generate_vocab:
        logger.warn('Caution: The vocab generated might be different from others(e.g., pretained models)')
        pyvocab = list(set(all_py))
        gpvocab = list(set(all_gp))
        pyvocab.sort()
        gpvocab.sort(key=lambda x: pypinyin.pinyin(x, 0)[0][0])
        with open('py.vocab', 'wt') as f:
            f.write('\n'.join(pyvocab))
        with open('gp.vocab', 'wt') as f:
            f.write('\n'.join(gpvocab))

    os.makedirs(args.dst_folder, exist_ok=True)
    write_scp(f'{args.dst_folder}/wav.scp', wav_scp) # 写入wav的路径
    write_scp(f'{args.dst_folder}/py.scp', py_scp)  #写入拼音
    write_scp(f'{args.dst_folder}/gp.scp', gp_scp) # 中文
    write_scp(f'{args.dst_folder}/dur.scp', dur_scp)#每一个音符的持续时间
    write_scp(f'{args.dst_folder}/spk.scp', spk_scp)#每一个音符的说话人id
    write_scp(f'{args.dst_folder}/mel.scp', mel_scp)#mel位置.



#===================
print('训练代码开始')



import argparse
import os

import numpy as np
import torch
print('当前torch版本是',torch.__version__)
import yaml
from torch.optim import SGD, Adam
from torch.optim.lr_scheduler import CyclicLR, ReduceLROnPlateau
from torch.utils.data import BatchSampler, DataLoader, SubsetRandomSampler
from torch.utils.tensorboard import SummaryWriter

from mtts.datasets.dataset import Dataset, collate_fn
from mtts.loss import FS2Loss
from mtts.models.fs2_model import FastSpeech2
from mtts.optimizer import ScheduledOptim
from mtts.utils.logging import get_logger
from mtts.utils.utils import save_image

logger = get_logger(__file__)


class AverageMeter:
    def __init__(self):
        self.mel_loss_v = 0.0
        self.posnet_loss_v = 0.0
        self.d_loss_v = 0.0
        self.total_loss_v = 0.0

        self._i = 0

    def update(self, mel_loss, posnet_loss, d_loss, total_loss):
        self.mel_loss_v = ((self.mel_loss_v * self._i) + mel_loss.item()) / (self._i + 1)
        self.posnet_loss_v = ((self.posnet_loss_v * self._i) + posnet_loss.item()) / (self._i + 1)
        self.d_loss_v = ((self.d_loss_v * self._i) + d_loss.item()) / (self._i + 1)
        self.total_loss_v = ((self.total_loss_v * self._i) + total_loss.item()) / (self._i + 1)

        self._i += 1
        return self.mel_loss_v, self.posnet_loss_v, self.d_loss_v, self.total_loss_v


def split_batch(data, i, n_split):
    n = data[1].shape[0]
    k = n // n_split
    ds = [d[:, i * k:(i + 1) * k] if j == 0 else d[i * k:(i + 1) * k] for j, d in enumerate(data)]
    return ds


def shuffle(data):
    n = data[1].shape[0]  # 输入的data 的物理含义:  token_tensors, durations, mels, torch.tensor(seq_len), torch.tensor(mel_len)    需要每一个解释一下. token_tensors (3,2,10): 3是因为 pinyin, 汉语, 说话人id 3个维度. 2是batch_size 10是句子长度. 然后是持续时间, 然后是mel序列.  都是(2,10) 然后后面2个是长度 都是 长度为2的数组而已.
    idx = np.random.permutation(n)
    data_shuffled = [d[:, idx] if i == 0 else d[idx] for i, d in enumerate(data)]
    return data_shuffled


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-r', '--restore', type=str, default='')
    parser.add_argument('-c', '--config', type=str, default='./config.yaml')
    parser.add_argument('-d', '--device', type=str, default='cuda')
    args = parser.parse_args()

    args.config='aishell3/config.yaml'
    args.restore='checkpoint_1350000.pth.tar'

    args.epoch=100 #一共训练多少轮.
    # args.device='cpu'

    device = args.device
    logger.info(f'using device {device}')

    with open(args.config) as f:
        config = yaml.safe_load(f)
        logger.info(f.read())

    dataset = Dataset(config)
    config['training']['batch_size']=2 ##########################为了debug时候加速
    dataloader = DataLoader(dataset,
                            batch_size=config['training']['batch_size'],
                            shuffle=False,
                            collate_fn=collate_fn,
                            drop_last=False,
                            num_workers=config['training']['num_workers'])

    step_per_epoch = len(dataloader) * config['training']['batch_size']

    model = FastSpeech2(config)
    model = model.to(args.device)
    #model.encoder.emb_layers.to(device) # ?

    optim_conf = config['optimizer']
    optim_class = eval(optim_conf['type'])
    logger.info(optim_conf['params'])
    optimizer = optim_class(model.parameters(), **optim_conf['params'])

    if args.restore != '':
        logger.info(f'Loading checkpoint {args.restore}')
        content = torch.load(args.restore)
        # model.load_state_dict(content['model'])
        model.load_state_dict(content)
        # optimizer.load_state_dict(content['optimizer'])
        # current_step = content['step']
        # start_epoch = current_step // step_per_epoch
        start_epoch = 0
        current_step = 0
        logger.info(f'loaded checkpoint at step {0}, epoch {start_epoch}')
    else:
        current_step = 0
        start_epoch = 0
        logger.info(f'Start training from scratch,step={current_step},epoch={start_epoch}')

    lrs = np.linspace(0, optim_conf['params']['lr'], optim_conf['n_warm_up_step'])
    Scheduler = eval(config['lr_scheduler']['type'])
    lr_scheduler = Scheduler(optimizer, **config['lr_scheduler']['params'])

    loss_fn = FS2Loss().to(device)
    train_logger = SummaryWriter(config['training']['log_path'])
    val_logger = SummaryWriter(config['training']['log_path'])
    avg = AverageMeter()
    for epoch in range(start_epoch, args.epoch):

        model.train()
        for i, data in enumerate(dataloader):   #首先我们分析data这个数据,  这行进行debug
            data = shuffle(data)
            max_src_len = torch.max(data[-2])
            max_mel_len = torch.max(data[-1])
            if 1:
            # for k in range(config['training']['batch_split']):
                if 0:
                  data_split = split_batch(data, k, config['training']['batch_split'])
                data_split=data
                tokens, duration, mel_truth, seq_len, mel_len = data_split
                #print(mel_len)
                tokens = tokens.to(device)
                duration = duration.to(device)
                mel_truth = mel_truth.to(device)
                seq_len = seq_len.to(device)
                mel_len = mel_len.to(device)
                # if torch.max(log_D) > 50:
                #  logger.info('skipping sample')
                #  continue

                mel_truth = mel_truth - config['fbank']['mel_mean']
                duration = duration - config['duration_predictor']['duration_mean']
                output = model(tokens, seq_len, mel_len, duration, max_src_len=max_src_len, max_mel_len=max_mel_len)

                mel_pred, mel_postnet, d_pred, src_mask, mel_mask, mel_len = output

                mel_loss, mel_postnet_loss, d_loss = loss_fn(d_pred, duration, mel_pred, mel_postnet, mel_truth,
                                                             ~src_mask, ~mel_mask)

                total_loss = mel_postnet_loss + d_loss + mel_loss
                ml, pl, dl, tl = avg.update(mel_loss, mel_postnet_loss, d_loss, total_loss)
                lr = optimizer.param_groups[0]['lr']
                if 0:
                    msg = f'epoch:{epoch},step:{current_step}|{step_per_epoch},loss:{tl:.3},mel:{ml:.3},'
                    msg += f'mel_postnet:{pl:.3},duration:{dl:.3},{lr:.3}'

                    if current_step % config['training']['log_step'] == 0:
                        logger.info(msg)
                print(current_step,'当前的loss是',total_loss)

                total_loss = total_loss / config['training']['acc_step']
                total_loss.backward()
                if current_step % config['training']['acc_step'] != 0:
                    continue

                current_step += 1
                #下面是调整学习率的.
                if current_step < config['optimizer']['n_warm_up_step']:
                    lr = lrs[current_step]
                    optimizer.param_groups[0]['lr'] = lr
                    optimizer.step()
                    optimizer.zero_grad()
                else:
                    optimizer.step()
                    lr_scheduler.step()
                    optimizer.zero_grad()

                if current_step % config['training']['synth_step'] == 0:
                    mel_pred = mel_pred.detach().cpu().numpy()
                    mel_truth = mel_truth.detach().cpu().numpy()
                    saved_path = os.path.join(config['training']['log_path'], f'{current_step}.png')
                    save_image(mel_truth[0][:mel_len[0]], mel_pred[0][:mel_len[0]], saved_path)
                    np.save(saved_path + '.npy', mel_pred[0])

                if current_step % config['training']['log_step'] == 0:

                    train_logger.add_scalar('total_loss', tl, current_step)
                    train_logger.add_scalar('mel_loss', ml, current_step)
                    train_logger.add_scalar('mel_postnet_loss', pl, current_step)
                    train_logger.add_scalar('duration_loss', dl, current_step)

                if 0:
                    if not os.path.exists(config['training']['checkpoint_path']):
                        os.makedirs(config['training']['checkpoint_path'])
                    ckpt_file = os.path.join(config['training']['checkpoint_path'],
                                             'checkpoint_{}.pth.tar'.format(current_step))
                    content = {'model': model.state_dict(), 'optimizer': optimizer.state_dict(), 'step': current_step}

                    torch.save(content, ckpt_file)
                    logger.info(f'Saved model at step {current_step} to {ckpt_file}')
    print(f"End of training for epoch {config['training']['epochs']}")
    print("saving!!!!!!!!!!!!!!!!!!!!!!!!!")
    if not os.path.exists(config['training']['checkpoint_path']):
        os.makedirs(config['training']['checkpoint_path'])
    ckpt_file = os.path.join(config['training']['checkpoint_path'],
                             'checkpoint_{}.pth.tar'.format(current_step))
    content = {'model': model.state_dict(), 'optimizer': optimizer.state_dict(), 'step': current_step}

    torch.save(content, ckpt_file)
    print(f'Saved model at step {current_step} to {ckpt_file}')
