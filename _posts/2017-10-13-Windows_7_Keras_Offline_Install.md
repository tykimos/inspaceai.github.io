---
layout: post
title: "Windows 7에서 텐서플로우, 케라스 오프라인 설치 2"
author: "정한솔"
date: 2017-10-13 14:00:00
categories: Lecture
comments: true
---

본 포스트에서는 Windows 10 환경에서 텐서플로우, 케라스를 오프라인으로 설치하는 방법에 대해서 설명하겠습니다. 보안 문제로 망분리된 서버나 워크스테이션에서 딥러닝 모델을 사용하려면 오프라인 설치가 필요합니다. '설치 준비' 파트에서 설치 파일을 다운로드 받고 '설치 방법' 파트의 설명을 따라 순서대로 설치하시면 됩니다.

---

### 설치 준비

필요한 오프라인 파일을 미리 다운로드 받아 준비하여야 합니다. 고용량이라 파일 다운로드 링크는 아래 댓글 창에 이메일을 남겨주시면 보내드리도록 하겠습니다.

 * <u>keras_offline_win7.zip</u> (1.8GB)

설치 파일의 구성 요소는 다음과 같습니다.

 * CUDA Toolkit 8.0
 * cuDNN 6.0 for CUDA 8.0
 * 아래 conda 패키지가 미리 포함된 Miniconda 인스톨러 ([conda/constructor](https://github.com/conda/constructor)를 사용하였습니다)
 * conda 패키지
   * python 3.5
   * pip
   * scikit-learn
   * pandas
   * matplotlib
   * jupyter notebook
 * pip 패키지
   * tensorflow
   * tensorflow-gpu
   * keras
   * opencv-python
   * h5py
   * pydot
 
---

### 설치 방법

C 드라이브 아래에 Projects 폴더를 생성하고 Projects 폴더 아래에 anaconda\_keras 폴더를 생성합니다. 이 폴더를 작업용 폴더로 사용하겠습니다. (C:\Projects\anaconda\_keras) 생성된 작업용 폴더 아래에 keras\_offline\_win7.zip 파일을 복사하여 압축을 해제합니다. 다음과 같은 파일이 들어 있습니다.

 * C:\Projects\anaconda\_keras\keras\_offline\_win7\InspaceDllab-1.0.0-Windows-x86_64.exe
 * C:\Projects\anaconda\_keras\keras\_offline\_win7\pip-packages.zip
 * C:\Projects\anaconda\_keras\keras\_offline\_win7\cuda\_8.0.61.2\_windows.exe
 * C:\Projects\anaconda\_keras\keras\_offline\_win7\cuda\_8.0.61\_windows.exe
 * C:\Projects\anaconda\_keras\keras\_offline\_win7\cudnn-8.0-windows7-x64-v6.0.zip

#### 필수 프로그램 설치

CUDA Toolkit, cuDNN, Miniconda를 컴퓨터에 설치하겠습니다. CPU 전용 Tensorflow를 설치하고자 하는 경우 CUDA Toolkit, cuDNN 설치는 건너뛰셔도 됩니다.

 * CUDA Toolkit 설치
   1. cuda\_8.0.61\_windows.exe을 실행하여 CUDA Toolkit을 설치합니다.
   2. cuda\_8.0.61.2\_windows.exe를 실행하여 CUDA Toolkit 업데이트를 추가 설치합니다.

|원본경로|대상경로|
|:-:|:-:|
|\cuda\bin|C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v8.0\bin|
|\cuda\include|C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v8.0\include|
|\cuda\lib\x64|C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v8.0\lib\x64|

 * cuDNN 설치
   1. cudnn-8.0-windows7-x64-v6.0.zip 을 압축 해제하고, 압축 해제한 폴더에서 위 표와 같이 파일을 복사합니다.

![anaconda_install_type]({{ "/images/2017-10-12-anaconda.png" | prepend: site.baseurl }})

 * Anaconda 설치 
   1. InspaceDllab-1.0.0-Windows-x86_64.exe를 실행하여 Miniconda를 설치합니다. 설치 타입은 Just Me (recommended)를 선택합니다. 그리고 2개의 체크박스가 있는 창이 나오는데 미니콘다 바이너리 폴더를 PATH에 추가할지의 여부와 미니콘다를 시스템 기본 파이썬으로 설정할지의 여부를 묻는 체크박스입니다. 둘다 체크한 상태로 두고 (기본값) 넘어갑니다.

### 추가 패키지 설치

우선 anaconda\_keras 폴더 아래의 pip-packages.zip 파일의 압축을 해제합니다. 그리고 시작 메뉴에서 "cmd"를 입력하여 명령 프롬프트를 실행합니다. 그리고 실행된 콘솔 창에 아래의 명령어를 입력합니다.

```
C:\Users\user> cd "C:\Projects\anaconda_keras\keras_offline_win10\conda-pip-packages"
C:\Projects\anaconda_keras\keras_offline_win10\pip-packages> pip install --no-index --find-links=. tensorflow-gpu keras opencv-python h5py pydot
```

만약 CPU 전용 텐서플로우를 사용하기를 원하는 경우 마지막 명령어 한 줄 대신 다음 명령어를 입력합니다.

```
C:\Projects\anaconda_keras\keras_offline_win10\pip-packages> pip install --no-index --find-links=. tensorflow keras opencv-python h5py pydot
```

tensorflow 패키지 이름에서 -gpu가 사라졌음을 주의하시기 바랍니다.

### 설치 테스트 및 문제 해결

Python 인터프리터를 실행하여 Keras를 임포트해 봅니다. Using TensorFlow backend. 라고 출력되면 성공입니다.

```
C:\Projects\anaconda_keras\keras_offline_win10\conda-pip-packages> python
>>> import keras
Using TensorFlow backend.
```

![testkeras]({{ "/images/2017-09-04-python.png" | prepend: site.baseurl }})

설치 후에 import keras를 할때 NumPy 관련 오류가 나는 경우가 있습니다. 그런 경우 pip 폴더에서 다음 명령어를 입력해 NumPy를 재설치 해 보시기 바랍니다. (성선경님 감사합니다)

```
pip install numpy-1.13.1-cp35-none-win_amd64.whl
```
