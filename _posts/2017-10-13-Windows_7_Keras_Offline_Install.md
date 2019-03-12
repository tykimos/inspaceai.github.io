---
layout: post
title: "Windows 7에서 텐서플로우, 케라스 오프라인 설치"
author: "정한솔"
date: 2017-10-13 14:00:00
categories: Lecture
comments: true
---

본 포스트에서는 Windows 7 환경에서 텐서플로우, 케라스를 오프라인으로 설치하는 방법에 대해서 설명하겠습니다. 보안 문제로 망분리된 서버나 워크스테이션에서 딥러닝 모델을 사용하려면 오프라인 설치가 필요합니다. '설치 준비' 파트에서 설치 파일을 다운로드 받고 '설치 방법' 파트의 설명을 따라 순서대로 설치하시면 됩니다.

 * 2018/2/20 설치 방법을 더 간편하게 변경하였고 각종 패키지의 버전을 tensorflow 1.5 등 최신 버전으로 업데이트하였습니다.

---

### 설치 준비

필요한 오프라인 파일을 미리 다운로드 받아 준비하여야 합니다. 아래 링크를 클릭해서 다운로드 받으시기 바랍니다. (원드라이브 링크로 연결됩니다)

 * [keras_offline_win7.zip](https://1drv.ms/u/s!AtbRowIzP4wEhPhJiJCJ7-8pBxJjeg) (1.75GB)

설치 파일의 구성 요소는 다음과 같습니다.

 * CUDA Toolkit 9.0
 * cuDNN 7.0 for CUDA 9.0
 * Python 3.6.4
 * Visual C++ 2010 런타임 라이브러리
 * pip 패키지
   * tensorflow
   * tensorflow-gpu
   * keras
   * opencv-python
   * h5py
   * pydot
   * pillow
   * matplotlib
   * jupyter
 
---

### 설치 방법

C 드라이브 아래에 Projects 폴더를 생성하고 Projects 폴더 아래에 anaconda\_keras 폴더를 생성합니다. 이 폴더를 작업용 폴더로 사용하겠습니다. (C:\Projects\anaconda\_keras) 생성된 작업용 폴더 아래에 keras\_offline\_win7.zip 파일을 복사하여 압축을 해제합니다. 다음과 같은 파일이 들어 있습니다.

 * C:\Projects\anaconda\_keras\keras\_offline\_win10\pip-packages.zip
 * C:\Projects\anaconda\_keras\keras\_offline\_win10\cuda\_9.0.176.1\_windows.exe
 * C:\Projects\anaconda\_keras\keras\_offline\_win10\cuda\_9.0.176\_windows.exe
 * C:\Projects\anaconda\_keras\keras\_offline\_win10\cudnn-9.0-windows7-x64-v7.zip
 * C:\Projects\anaconda\_keras\keras\_offline\_win10\python-3.6.4-amd64.exe
 * C:\Projects\anaconda\_keras\keras\_offline\_win10\vcredist_x64.exe

#### 필수 프로그램 설치

CUDA Toolkit, cuDNN, Python을 컴퓨터에 설치하겠습니다. CPU 전용 Tensorflow를 설치하고자 하는 경우 CUDA Toolkit, cuDNN 설치는 건너뛰셔도 됩니다. 엔비디아 그래픽을 사용하는 환경이 아닌 경우 반드시 CPU 전용 Tensorflow를 설치하셔야 합니다.

 * CUDA Toolkit 설치
   1. cuda\_9.0.176\_windows.exe을 실행하여 CUDA Toolkit을 설치합니다.
   2. cuda\_9.0.176.1\_windows.exe를 실행하여 CUDA Toolkit 업데이트를 추가 설치합니다.

|원본경로|대상경로|
|:-:|:-:|
|\cuda\bin|C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v9.0\bin|
|\cuda\include|C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v9.0\include|
|\cuda\lib\x64|C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v9.0\lib\x64|

 * cuDNN 설치
   1. cudnn-9.0-windows7-x64-v7.zip 을 압축 해제하고, 압축 해제한 폴더에서 위 표와 같이 파일을 복사합니다.

![python installer]({{ "/images/2017-10-12-python-installer.png" | prepend: site.baseurl }})

 * Python 설치
   1. python-3.6.4-amd64.exe를 실행하여 파이썬을 설치합니다. 처음 설치 창에서 Add Python 3.6 to PATH를 체크하고 진행합니다.

### 추가 패키지 설치

matplotlib 라이브러리를 사용하기 위해서는 Visual C++ 2010 런타임 라이브러리가 필요합니다. vcredist_x64.exe를 실행하여 설치합니다. 이미 설치되어 있는 경우 설치가 진행되지 않을 수 있으며 이 경우는 그냥 넘어가시면 됩니다. 그 후, anaconda\_keras 폴더 아래의 pip-packages.zip 파일의 압축을 해제합니다. 그리고 시작 메뉴에서 "cmd"를 입력하여 명령 프롬프트를 실행합니다. 그리고 실행된 콘솔 창에 아래의 명령어를 입력합니다.

```
C:\Users\user> cd "C:\Projects\anaconda_keras\keras_offline_win7\pip-packages"
C:\Projects\anaconda_keras\keras_offline_win7\pip-packages> python -m pip install --no-index --find-links=. jupyter matplotlib tensorflow-gpu keras opencv-python h5py pydot pillow
```

만약 CPU 전용 텐서플로우를 사용하기를 원하는 경우 마지막 명령어 대신 다음 명령어를 입력합니다.

```
C:\Projects\anaconda_keras\keras_offline_win7\pip-packages> python -m pip install --no-index --find-links=. jupyter matplotlib tensorflow keras opencv-python h5py pydot pillow
```

tensorflow 패키지 이름에서 -gpu가 사라졌음을 주의하시기 바랍니다.

### 설치 테스트 및 문제 해결

Python 인터프리터를 실행하여 Keras를 임포트해 봅니다. Using TensorFlow backend. 라고 출력되면 성공입니다.

```
C:\Projects\anaconda_keras\keras_offline_win7\conda-pip-packages> python
>>> import keras
Using TensorFlow backend.
```

![testkeras]({{ "/images/2017-10-12-python.png" | prepend: site.baseurl }})
