# Python_environment_setting
파이썬 관련 각종 환경설정


# [가상환경]

## 기타 가상환경 설정

### 삭제

jupyter kernelspec uninstall 이름 > ipykernel 삭제

conda env remove -n 이름 > env 삭제


## Pytorch 환결설정 예시 (설치 당시 기준)
- 가상환경명 : torch
- python 3.8.5 / pytorch 1.8.1 - cuda 10.1 cudnn7

conda create -n torch python anaconda

conda install pytorch torchvision torchaudio cudatoolkit=10.1 -c pytorch

### cuda 사용가능 여부 확인

import torch

USE_CUDA = torch.cuda.is_available()

print(USE_CUDA)

device = torch.device('cuda:0' if USE_CUDA else 'cpu')

print('학습을 진행하는 기기:',device)



### pytorch 홈페이지 설치 url
- https://pytorch.org/

- 상태 선택하면되는데 10.1 cuda 기준이 없어서 숫자 바꿔서 진행했었음


## Keras 환경설정 예시 
- 가상환경명 : tf_keras
- tf-gpu 2.3.0 / keras 2.3.1
conda create -n tf_keras python=3.8.5 anaconda

conda activate tf_keras

pip install tensorflow-gpu==2.3.0

( tf 버전 확인 >> python -> import tensorflow as tf -> print(tf.__version__) -> exit() )

pip install ipykernel

python -m ipykernel install --user --name tf_keras --display-name "tf_keras"

pip install keras==2.3.1

### ipykernel 주의점
- install 및 ipykernel 생성할때 가상환경 접속(conda activate 이름) 이후에 진행해야함
- 현재 가상환경 되어있는 곳에 대한 ipykernel이 생성되는 것이기 때문임

### keras 설치시 [ "keras" vs "keras-gpu" ]
- tf-gpu 설치 이후에 keras-gpu를 같이 설치하면 충돌이 날 수 있다고 봤었음
- 그래서 keras-gpu 말고 그냥 keras 설치
