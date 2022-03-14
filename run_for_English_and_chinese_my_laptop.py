juzi='hello everyone 之前在微博 分享过一个英语句子I am because you are.然后有同学说这不是一个错句吗'
juzi='hello everyone 之前在微博'
silence_between_sentence=0.7
#=抽取中文其他部分都给英文
import os
# os.environ['HOME']='E:/'
def _is_chinese_char(cp):
    """Checks whether CP is the codepoint of a CJK character."""
    # This defines a "chinese character" as anything in the CJK Unicode block:
    #   https://en.wikipedia.org/wiki/CJK_Unified_Ideographs_(Unicode_block)
    #
    # Note that the CJK Unicode block is NOT all Japanese and Korean characters,
    # despite its name. The modern Korean Hangul alphabet is a different block,
    # as is Japanese Hiragana and Katakana. Those alphabets are used to write
    # space-separated words, so they are not treated specially and handled
    # like the all of the other languages.
    if (
            (cp >= 0x4E00 and cp <= 0x9FFF)
            or (cp >= 0x3400 and cp <= 0x4DBF)  #
            or (cp >= 0x20000 and cp <= 0x2A6DF)  #
            or (cp >= 0x2A700 and cp <= 0x2B73F)  #
            or (cp >= 0x2B740 and cp <= 0x2B81F)  #
            or (cp >= 0x2B820 and cp <= 0x2CEAF)  #
            or (cp >= 0xF900 and cp <= 0xFAFF)
            or (cp >= 0x2F800 and cp <= 0x2FA1F)  #
    ):  #
        return True

    return False
chinese=''.join([i  if _is_chinese_char(ord(i)) else ' ' for i in juzi ])
english=''.join([i  if not _is_chinese_char(ord(i)) else ' ' for i in juzi ])
#首先我们进行中英文拆分.
#==================首先合成英文部分
a=chinese.split(' ')
b=english.split(' ')
flag=[] #中文标1,英文标0

for i in chinese:
    if i!=' ':
        flag.append(1)
    else:
        flag.append(0)
print(1)


#

#========算法思路就是每次取一个相同flag的片段, 然后把它变成wav信号.
start=0
end=0
out=[]
for i in range(1,len(flag)):#========这段逻辑还是有点复杂,需要考虑最后的边界.



    if flag[i]!=flag[i-1] :
        out.append([juzi[start:i],flag[start]])
        # out.append([start,i,flag[start]])
        start=i


    if i == len(flag) - 1:
        out.append([juzi[start:i + 1], flag[start]])

out_juzi=out
print(1)




#=分割成数组.

print(1)


# 英文懒得弄了.直接 torch hub找一个试试
# this assumes that you have a proper version of PyTorch already installed
# pip install -q torchaudio omegaconf
# https://github.com/snakers4/silero-models  文件地址 可以手动下载项目里面的model.yml即可. 现在我已经下好了.直接运行即可!!!!!!!!!
#=============我知道了,输入的句子不能太长. 因为训练预料时候句子就这么短. 大概50个词左右,太长了,句子会自动截断!



import torch

language = 'en'
speaker = 'lj_16khz'
device = torch.device('cpu')
remote='snakers4/silero-models'
bendi=''
hub_dir='data_for_hub'
model, symbols, sample_rate, example_text, apply_tts = torch.hub.load(repo_or_dir='snakers4/silero-models',
                                                                      model='silero_tts',
                                                                      language=language,
                                                                      speaker=speaker)
english_sr=sample_rate
example_text=juzi
model_english = model.to(device)  # gpu or cpu
# audio = apply_tts(texts=[example_text],
#                   model=model,
#                   sample_rate=sample_rate,
#                   symbols=symbols,
#                   device=device)


import argparse
import os
import subprocess

import numpy as np
import torch
import yaml
from scipy.io import wavfile
# audio=audio[0].numpy()
# wavfile.write('aaa.wav', sample_rate, audio)
# print(1)



#===========chinese syn










with open('gp.vocab',encoding='utf-8') as f:
    tmp=[i.strip() for i in f.readlines()]



#============过滤掉字典中没有的东西.
# ttttttt=''.join([i for i in ttttttt if i in tmp])
# print(1)

import argparse
import os
import subprocess

import numpy as np
import torch
import yaml
from scipy.io import wavfile

from mtts.models.fs2_model import FastSpeech2
from mtts.models.vocoder import *
from mtts.text import TextProcessor
from mtts.utils.logging import get_logger

logger = get_logger(__file__)


def check_ffmpeg():
    r, path = subprocess.getstatusoutput("which ffmpeg")
    return r == 0


with_ffmpeg = check_ffmpeg()


def build_vocoder(device, config):
    vocoder_name = config['vocoder']['type']
    VocoderClass = eval(vocoder_name)
    del config['vocoder'][vocoder_name]['checkpoint']
    model = VocoderClass(**config['vocoder'][vocoder_name])
    return model


def normalize(wav):
    assert wav.dtype == np.float32
    eps = 1e-6
    sil = wav[1500:2000]
    #wav = wav - np.mean(sil)
    #wav = (wav - np.min(wav))/(np.max(wav)-np.min(wav)+eps)
    wav = wav / np.max(np.abs(wav))
    #wav = wav*2-1
    wav = wav * 32767
    return wav.astype('int16')


def to_int16(wav):
    wav = wav = wav * 32767
    wav = np.clamp(wav, -32767, 32768)
    return wav.astype('int16')






print('生成需要的汉字拼音')
import argparse
import copy
import os
from typing import List

import jieba
import pypinyin




def read_vocab(file: os.PathLike) -> List[str]:
    with open(file,encoding='utf-8') as f:
        vocab = f.read().split('\n')
        vocab = [v for v in vocab if len(v) > 0 and v != '\n']
    return vocab

SPECIAL_NOTES = '。？！?!.;；:,，:'
class TextNormal:
    SPECIAL_NOTES = '。？！?!.;；:,，:'
    def __init__(self,
                 gp_vocab_file: os.PathLike,
                 py_vocab_file: os.PathLike,
                 add_sp1=False,
                 fix_er=False,
                 add_sil=True):
        if gp_vocab_file is not None:
            self.gp_vocab = read_vocab(gp_vocab_file)
        if py_vocab_file is not None:
            self.py_vocab = read_vocab(py_vocab_file)
            self.in_py_vocab = dict([(p, True) for p in self.py_vocab])
        self.add_sp1 = add_sp1
        self.add_sil = add_sil
        self.fix_er = fix_er

        # gp2idx = dict([(c, i) for i, c in enumerate(self.gp_vocab)])
        # idx2gp = dict([(i, c) for i, c in enumerate(self.gp_vocab)])

    def _split2sent(self, text):
        new_sub = [text]
        while True:
            sub = copy.deepcopy(new_sub)
            new_sub = []
            for s in sub:
                sp = False
                for t in SPECIAL_NOTES:
                    if t in s:
                        new_sub += s.split(t)
                        sp = True
                        break

                if not sp and len(s) > 0:
                    new_sub += [s]
            if len(new_sub) == len(sub):
                break
        tokens = [a for a in text if a in SPECIAL_NOTES]

        return new_sub, tokens

    def _correct_tone3(self, pys: List[str]) -> List[str]:
        """Fix the continuous tone3 pronunciation problem"""
        for i in range(2, len(pys)):
            if pys[i][-1] == '3' and pys[i - 1][-1] == '3' and pys[i - 2][-1] == '3':
                pys[i - 1] = pys[i - 1][:-1] + '2'  # change the middle one
        for i in range(1, len(pys)):
            if pys[i][-1] == '3':
                if pys[i - 1][-1] == '3':
                    pys[i - 1] = pys[i - 1][:-1] + '2'
        return pys

    def _correct_tone4(self, pys: List[str]) -> List[str]:
        """Fixed the problem of pronouncing 不 bu2 yao4 / bu4 neng2"""
        for i in range(len(pys) - 1):
            if pys[i] == 'bu4':
                if pys[i + 1][-1] == '4':
                    pys[i] = 'bu2'
        return pys

    def _replace_with_sp(self, pys: List[str]) -> List[str]:
        for i, p in enumerate(pys):
            if p in ',，、':
                pys[i] = 'sp1'
        return pys

    def _correct_tone5(self, pys: List[str]) -> List[str]:
        for i in range(len(pys)):
            if pys[i][-1] not in '1234':
                pys[i] += '5'
        return pys

    def gp2py(self, gp_text: str) -> List[str]:

        gp_sent_list, tokens = self._split2sent(gp_text)
        py_sent_list = []
        for sent in gp_sent_list:
            pys = []
            for words in list(jieba.cut(sent)):
                py = pypinyin.pinyin(words, pypinyin.TONE3)
                py = [p[0] for p in py]
                pys += py
            if self.add_sp1:
                pys = self._replace_with_sp(pys)
            pys = self._correct_tone3(pys)
            pys = self._correct_tone4(pys)
            pys = self._correct_tone5(pys)
            if self.add_sil:
                py_sent_list += [' '.join(['sil'] + pys + ['sil'])]
            else:
                py_sent_list += [' '.join(pys)]

        if self.add_sil:
            gp_sent_list = ['sil ' + ' '.join(list(gp)) + ' sil' for gp in gp_sent_list]
        else:
            gp_sent_list = [' '.join(list(gp)) for gp in gp_sent_list]

        if self.fix_er:
            new_py_sent_list = []
            for py, gp in zip(py_sent_list, gp_sent_list):
                py = self._convert_er2(py, gp)
                new_py_sent_list += [py]
            py_sent_list = new_py_sent_list
            # print(new_py_sent_list)

        return py_sent_list, gp_sent_list

    def _convert_er2(self, py, gp):
        py2hz = dict([(p, h) for p, h in zip(py.split(), gp.split())])
        py_list = py.split()
        for i, p in enumerate(py_list):
            if (p == 'er2' and py2hz[p] == '儿' and i > 1 and len(py_list[i - 1]) > 2 and py_list[i - 1][-1] in '1234'):

                py_er = py_list[i - 1][:-1] + 'r' + py_list[i - 1][-1]

                if self.in_py_vocab.get(py_er, False):  # must in vocab
                    py_list[i - 1] = py_er
                    py_list[i] = 'r'
        py = ' '.join(py_list)
        return py






if 1:
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', type=str, default='input.txt')
    parser.add_argument('--duration', type=float, default=1.0)
    parser.add_argument('--output_dir', type=str, default='./outputs/')
    parser.add_argument('--checkpoint', type=str, required=False, default='')
    parser.add_argument('-c', '--config', type=str, default='./config.yaml')
    parser.add_argument('-d', '--device', choices=['cuda', 'cpu'], type=str, default='cpu')
    args = parser.parse_args()
    args.checkpoint='E:/checkpoint_1350000.pth.tar'
    args.config = 'aishell3/config.yaml'
    args.input='test.txt'
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    with open(args.config) as f:
        config = yaml.safe_load(f)
        logger.info(f.read())

    sr = config['fbank']['sample_rate']
    os.environ['HOME']='C:/'
    vocoder = build_vocoder(args.device, config)
    text_processor = TextProcessor(config)
    model = FastSpeech2(config)

    if args.checkpoint != '':
        sd = torch.load(args.checkpoint, map_location=args.device)
        if 'model' in sd.keys():
            sd = sd['model']
    model.load_state_dict(sd)
    del sd  # to save mem
    model = model.to(args.device)
    torch.set_grad_enabled(False)

    try:
        lines = open(args.input).read().split('\n')
    except:
        print('Failed to open text file', args.input)
        print('Treating input as text')
        lines = [args.input]
    out=[]


    import time

    # 打印时间戳
    # print(time.time())  # 打印自从1970年1月1日午夜（历元）经过了多长时间，以秒为单位
    # 打印本地时间
    # print(time.localtime(time.time()))  # 打印本地时间
    # 打印格式化时间
    a=time.strftime('%Y-%m-%d-%H:%M:%S', time.localtime(time.time()))  # 打印按指定格式排版的时间
    print(a)
    # a=time.time_ns()
    # args.text=ttttttt





def syn_chinese(txt):
    global args
    ttttttt = txt
    speakerid = 100
    silence_between_sentence = 0.7  # 逗号句号问号这些符号统一的暂停时间.
    SPECIAL_NOTES = '。？！?!.;；:,，:'
    text = ttttttt




    tn = TextNormal('gp.vocab', 'py.vocab', add_sp1=True, fix_er=True)
    text=text.replace('\n','')
    text=text.replace(' ','')
    py_list, gp_list = tn.gp2py(text)


    kkk=[]
    for py, gp in zip(py_list, gp_list):
       out=(py + '|' + gp)

       kkk2=str(a)+'|'+out+'|'+str(speakerid)
       kkk.append(kkk2)
    print(1)















    out=[]
    for line in kkk:


        print(line,33333333333333333333333333333333)
        name, tokens = text_processor(line)
        tokens = tokens.to(args.device)
        seq_len = torch.tensor([tokens.shape[1]])
        tokens = tokens.unsqueeze(1)
        seq_len = seq_len.to(args.device)
        max_src_len = torch.max(seq_len)
        output = model(tokens, seq_len, max_src_len=max_src_len, d_control=args.duration)
        mel_pred, mel_postnet, d_pred, src_mask, mel_mask, mel_len = output

        # convert to waveform using vocoder
        mel_postnet = mel_postnet[0].transpose(0, 1).detach()   # mel_postnet 是我们真正需要的东西!!!!!!!!!
        mel_postnet += config['fbank']['mel_mean']
        wav = vocoder(mel_postnet)
        out.append(wav)
        import numpy
    wav = numpy.concatenate(out, axis=0)
    if 0:
        if out:
            wav=out[0]
            for i in range(len(out)-1):
                wav2=numpy.zeros([sr*silence_between_sentence]).astype('float32')
                wav=numpy.concatenate([wav,wav2],axis=0)
            wav = numpy.concatenate([wav, out[-1]], axis=0)
        # wav=numpy.concatenate(out,axis=0)
        # if 1:#根据自己的需要加入沉默时间
        #     wav2=numpy.zeros([sr*10]).astype('float32')
        #     wav=numpy.concatenate([wav,wav2],axis=0)








    if 1:
        if config['synthesis']['normalize']:
            wav = normalize(wav)
        else:
            wav = to_int16(wav)
        dst_file = os.path.join(args.output_dir, f'{name}.wav')
        return wav
        #np.save(dst_file+'.npy',mel_postnet.cpu().numpy())
        # logger.info(f'writing file to {dst_file}')
        # wavfile.write(dst_file, sr, wav)
import torchaudio
# syn_chinese()
#==========all     synthesis
import re
out_voice=[]
# out_juzi=[out_juzi[1],out_juzi[1]]
# out_juzi=out_juzi[::-1]
for i in out_juzi:
    if i[1]==0 and len(re.sub(r'\s+', ' ', i[0]).strip())>=1:

            print('running',i)
            aaa=apply_tts(texts=[i[0]],
                      model=model_english,
                      sample_rate=sample_rate,
                      symbols=symbols,
                      device=device)[0]
            print(aaa)
            aaa=torchaudio.transforms.Resample(orig_freq=english_sr, new_freq=sr)(aaa).numpy()
            import numpy
            out_voice.append(aaa )
            bbb=aaa
            # out_voice.append(  numpy.zeros(int(sr*silence_between_sentence)).astype('float32'))
            wavfile.write('tmp3.wav', sr, aaa)


    if i[1]==1:
            print('running', i) #从int16 转化为float32 搞定

            aaa=syn_chinese(i[0]).astype('float32')/32768
            # aaa/32768
            out_voice.append( aaa)
            wavfile.write('tmp2.wav', sr, aaa)

import pyaudio
pyaudio.paFloat32
print(1)
import numpy
wav = numpy.concatenate(out_voice, axis=0)
wavfile.write('tmp.wav', sr, wav)