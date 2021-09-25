# Python_environment_setting
파이썬 관련 각종 환경설정


# [가상환경]


## ■ 기타 가상환경 설정

### 삭제
```
jupyter kernelspec uninstall 이름 # ipykernel 삭제
conda env remove -n 이름 # env 삭제
```

## Pytorch 환결설정 예시 (설치 당시 기준)
- 가상환경명 : torch
- python 3.8.5 / pytorch 1.8.1 - cuda 10.1 cudnn7.6.5
```
conda create -n torch python=3.8.5 anaconda
conda activate torch

conda install pytorch torchvision torchaudio cudatoolkit=10.1 -c pytorch

pip install ipykernel
python -m ipykernel install --user --name torch --display-name "torch"
```
### cuda 사용가능 여부 확인
```
import torch
USE_CUDA = torch.cuda.is_available()
print(USE_CUDA)

device = torch.device('cuda:0' if USE_CUDA else 'cpu')
print('학습을 진행하는 기기:',device)
```


### pytorch 홈페이지 설치 url
- https://pytorch.org/

- 상태 선택하면되는데 10.1 cuda 기준이 없어서 숫자 바꿔서 진행했었음

## ■ Keras 환경설정 예시
- 가상환경명 : tf_keras
- tf-gpu 2.3.0 / keras 2.3.1
```
conda create -n tf_keras python=3.8.5 anaconda
conda activate tf_keras
pip install tensorflow-gpu==2.3.0
```
( tf 버전 확인 >> python -> import tensorflow as tf -> print(tf.__version__) -> exit() )
```
pip install ipykernel
python -m ipykernel install --user --name tf_keras --display-name "tf_keras"
pip install keras==2.3.1
```
### CUDA 사용가능 여부 확인
```
from tensorflow.python.client import device_lib
device_lib.list_local_devices()
```


### ipykernel 주의점
- install 및 ipykernel 생성할때 가상환경 접속(conda activate 이름) 이후에 진행해야함
- 현재 가상환경 되어있는 곳에 대한 ipykernel이 생성되는 것이기 때문임

### keras 설치시 [ "keras" vs "keras-gpu" ]
- tf-gpu 설치 이후에 keras-gpu를 같이 설치하면 충돌이 날 수 있다고 봤었음
- 그래서 keras-gpu 말고 그냥 keras 설치

---

# [Google Colab]


## ■ 24시간 유지되도록 웹 console에 입력
```
function ClickConnect(){
console.log("Working"); 
document
  .querySelector('#top-toolbar > colab-connect-button')
  .shadowRoot.querySelector('#connect')
  .click() 
}
setInterval(ClickConnect,60000)
```

# [CUDA]

## 설치 순서

### 0. Visual Studio 2019 설치

https://visualstudio.microsoft.com/ko/downloads/

 -> Community version
 
 -> 설치 중 C++ 개발환경만 설정하라고 나와있는데, Python도 같이 선택함

### 1. 설치 가능한 Cuda version 확인

https://docs.nvidia.com/deploy/cuda-compatibility/index.html#binary-compatibility

### 2. CUDA Toolkit 설치 (v10.1)

### 3. cuDNN 설치 (v7.6.5)

 -> 압축 해제 및 CUDA 설치 경로에 복사, 붙여넣기
 
 -> C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v10.1

### 4. 환경변수 설정

 -> 시스템 환경변수 설정 > 환경변수에서

  [1] 시스템 변수에 CUDA_PATH 2개 확인

  [2] 사용자 변수 Path에 위 cuDNN 복사한 경로 중 bin, lib, include에 대한 경로 추가

### 5. 설치 확인

-> cmd > nvcc --version

![image](https://user-images.githubusercontent.com/67678405/119101269-c1b74f00-ba53-11eb-9a98-f09effca5578.png)


# [기타 모듈]

## ■ 크롬 웹 드라이버

#### 1. 크롬 웹 드라이버 버전에 맞게 py파일과 같은 path에 다운로드 해야함.
#### 2. 크롬 업데이트시 크롬 웹 드라이버도 버전에 맞게 업데이트 해야함.
#### 3. 크롬 버전은 설정 - Chrome 정보에서 확인 가능
#### 4. https://chromedriver.chromium.org/downloads (드라이버 다운로드 위치)

