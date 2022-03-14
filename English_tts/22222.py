# 英文懒得弄了.直接 torch hub找一个试试
# this assumes that you have a proper version of PyTorch already installed
# pip install -q torchaudio omegaconf
# https://github.com/snakers4/silero-models  文件地址 可以手动下载项目里面的model.yml即可. 现在我已经下好了.直接运行即可!!!!!!!!!
#=============我知道了,输入的句子不能太长. 因为训练预料时候句子就这么短. 大概50个词左右,太长了,句子会自动截断!



import torch

language = 'en'
speaker = 'lj_16khz'
device = torch.device('cpu')
model, symbols, sample_rate, example_text, apply_tts = torch.hub.load(repo_or_dir='snakers4/silero-models',
                                                                      model='silero_tts',
                                                                      language=language,
                                                                      speaker=speaker)
example_text='''

early 13c., "the Creed in the Church service," from Latin credo "I believe," the first word of the Apostles' and Nicene creeds, first person singular present indicative of credere "to believe," from PIE compound *kerd-dhe- "to believe," literally "to put one's heart" (source also of Old Irish cretim, Irish creidim, Welsh credu "I believe," Sanskrit śrad-dhā- "faith, confidence, devotion"), from PIE root *kerd- "heart." The nativized form is creed. General sense of "formula or statement of belief" is from 1580s.
'''
model = model.to(device)  # gpu or cpu
audio = apply_tts(texts=[example_text],
                  model=model,
                  sample_rate=sample_rate,
                  symbols=symbols,
                  device=device)


import argparse
import os
import subprocess

import numpy as np
import torch
import yaml
from scipy.io import wavfile
audio=audio[0].numpy()
wavfile.write('aaa.wav', sample_rate, audio)
print(1)