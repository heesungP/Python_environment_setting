# Python_environment_setting
파이썬 관련 각종 환경설정


# 가상환경

## Keras 환경설정 예시 - tf_keras라는 이름의 가상환경 (tf-gpu 2.3.0 / keras 2.3.1)
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
