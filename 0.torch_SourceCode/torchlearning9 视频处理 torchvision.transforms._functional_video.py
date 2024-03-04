# -*- coding: utf-8 -*-
"""
Created on Tue Feb 20 19:37:09 2024

@author: DELL
"""
import torchvision
'''
# 图片处理
# 类库
torchvision.transforms.transforms
# 函数库
torchvision.transforms.functional
'''

'''
# 视频处理
# 类库
torchvision.transforms._transforms_video
# 函数库
torchvision.transforms._functional_video
'''
import torchvision
#返回 图像、音频、帧率信息
framePicture,frameAudio,fps=torchvision.io.read_video('../CNN/x.mp4',pts_unit='sec')
#T*H*W*C
from torchvision.transforms import _functional_video
iarray=_functional_video.to_tensor(framePicture[:10])
#C*T*H*W
import torchvision.transforms.functional as F
x=framePicture[0]
x=x.permute(2,0,1)
F.to_pil_image(x)

#返回视频时间戳
pts,fps_=torchvision.io.read_video_timestamps('../CNN/video.mp4',pts_unit='sec')


#保存视频  
#346帧率图像以每秒4帧显示  人眼接受>25帧
#音频采样率48000  人耳接受48K/44K
framePicture2,frameAudio2,fps2=torchvision.io.read_video('../CNN/y.mp4',pts_unit='sec')
#音轨互换
torchvision.io.write_video('../CNN/xy.mp4',
                           video_array=framePicture,fps=fps['video_fps'],video_codec='h264',
                           audio_array=frameAudio2,audio_fps=fps['audio_fps'],audio_codec='aac')


'''
音频处理torchaudio
类库
torchaudio.transforms._transforms
函数库
torchaudio.functional.functional
'''
import torchaudio
#读入音频
# torchaudio.io.StreamReader()
#输出音频
# torchaudio.io.StreamWriter()






