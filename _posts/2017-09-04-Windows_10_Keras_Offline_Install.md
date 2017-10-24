---
layout: post
title: "Windows 10에서 텐서플로우, 케라스 오프라인 설치"
author: "정한솔"
date: 2017-09-04 12:00:00
categories: Lecture
comments: true
---

Jupyter Notebook 등의 추가 패키지를 포함한 오프라인 설치 방법을 새로 포스팅했습니다. 새로 포스팅된 [Windows 10에서 텐서플로우, 케라스 오프라인 설치 2]({{ "/lecture/2017/10/13/Windows_10_Keras_Offline_Install.html" | prepend: site.baseurl }})를 참조하시기 바랍니다.

---

본 포스트에서는 Windows 10 환경에서 텐서플로우, 케라스를 오프라인으로 설치하는 방법에 대해서 설명하겠습니다. 보안 문제로 망분리된 서버나 워크스테이션에서 딥러닝 모델을 사용하려면 오프라인 설치가 필요합니다. '설치 준비' 파트에서 설치 파일을 다운로드 받고 '설치 방법' 파트의 설명을 따라 순서대로 설치하시면 됩니다.

---

### 설치 준비

필요한 오프라인 파일을 미리 다운로드 받아 준비하여야 합니다. 고용량이라 파일 다운로드 링크는 아래 댓글 창에 이메일을 남겨주시면 보내드리도록 하겠습니다.

 * <u>keras_offline_win10.zip</u> (1.9GB)

설치 파일의 구성 요소는 다음과 같습니다.

 * CUDA Toolkit 8.0.61.2
 * cuDNN 6.0 for CUDA 8.0
 * Anaconda 4.4.0 for Python 3
 * conda 패키지
   * Python 3.5.4
   * pip 9.0.1
   * SciPy 0.19.1 등
 * pip 패키지
   * TensorFlow 1.3.0
   * Keras 2.0.8
   * NumPy 1.13.1 등
 
---

### 설치 방법

C 드라이브 아래에 Projects 폴더를 생성하고 Projects 폴더 아래에 anaconda\_keras 폴더를 생성합니다. 이 폴더를 작업용 폴더로 사용하겠습니다. (C:\Projects\anaconda\_keras) 생성된 작업용 폴더 아래에 keras\_offline\_win7.zip 또는 keras\_offline\_win10.zip 파일을 복사하여 압축을 해제합니다. 다음과 같은 파일이 들어 있습니다.

 * C:\Projects\anaconda\_keras\keras\_offline\_win10\Anaconda3-4.4.0-Windows-x86\_64.exe
 * C:\Projects\anaconda\_keras\keras\_offline\_win10\conda-pip-packages.zip
 * C:\Projects\anaconda\_keras\keras\_offline\_win10\cuda\_8.0.61.2\_windows.exe
 * C:\Projects\anaconda\_keras\keras\_offline\_win10\cuda\_8.0.61\_win10.exe
 * C:\Projects\anaconda\_keras\keras\_offline\_win10\cudnn-8.0-windows10-x64-v6.0.zip

#### 필수 프로그램 설치

CUDA Toolkit, cuDNN, Anaconda를 컴퓨터에 설치하겠습니다. CPU 전용 Tensorflow를 설치하고자 하는 경우 CUDA Toolkit, cuDNN 설치는 건너뛰셔도 됩니다. 또한 CUDA Toolkit, cuDNN의 경우 운영체제에 따라 설치 파일 이름이 다르므로 자신이 가지고 있는 파일 하나만 확인하시면 됩니다.

 * CUDA Toolkit 설치
   1. cuda\_8.0.61\_win10.exe을 실행하여 CUDA Toolkit을 설치합니다.
   2. cuda\_8.0.61.2\_windows.exe를 실행하여 CUDA Toolkit 업데이트를 추가 설치합니다.

|원본경로|대상경로|
|:-:|:-:|
|\cuda\bin|C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v8.0\bin|
|\cuda\include|C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v8.0\include|
|\cuda\lib\x64|C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v8.0\lib\x64|

 * cuDNN 설치
   1. cudnn-8.0-windows10-x64-v6.0.zip 을 압축 해제하고, 압축 해제한 폴더에서 위 표와 같이 파일을 복사합니다.

![anaconda_install_type]({{ "/images/2017-09-04-anaconda.png" | prepend: site.baseurl }})

 * Anaconda 설치 
   1. Anaconda3-4.4.0-Windows-x86\_64.exe를 실행하여 Anaconda를 설치합니다. 설치 타입은 Just Me (recommended)를 선택합니다.

### 가상 환경 생성 및 패키지 설치

시작 메뉴에서 "Anaconda Prompt"를 검색하여 실행합니다. 실행된 콘솔 창에 아래의 명령어를 차례대로 입력합니다.

```
(C:\Users\user\Anaconda3) C:\Users\user> conda clean --all
```

현재 캐시되어 있는 모든 패키지를 삭제합니다. y/n 입력에는 모두 y를 입력합니다.

```
(C:\Users\user\Anaconda3) C:\Users\user> conda create --name venv --clone root
(C:\Users\user\Anaconda3) C:\Users\user> activate venv
```

아무것도 설치되어 있지 않은 빈 환경 venv를 생성하고 활성화하였습니다. 생성 도중 오류가 몇개 발생하나 무시해도 됩니다. 그리고 conda-pip-packages.zip 파일의 압축을 해제합니다. conda-pip-packages 폴더 아래에 conda, pip 폴더 두가지가 생성됩니다. cd 명령어를 이용해 압축을 푼 폴더 안으로 이동합니다.

```
(venv) C:\Users\user> cd "C:\Projects\anaconda_keras\keras_offline_win10\conda-pip-packages"
```

운영체제에 따라 위 두 명령어 중 하나를 입력합니다.

```
(venv) C:\Projects\anaconda_keras\keras_offline_win10\conda-pip-packages> cd conda
(venv) C:\Projects\anaconda_keras\keras_offline_win10\conda-pip-packages\conda> conda install python-3.5.4-0.tar.bz2
(venv) C:\Projects\anaconda_keras\keras_offline_win10\conda-pip-packages\conda> conda install pip-9.0.1-py35_1.tar.bz2
(venv) C:\Projects\anaconda_keras\keras_offline_win10\conda-pip-packages\conda> conda install wheel-0.29.0-py35_0.tar.bz2
(venv) C:\Projects\anaconda_keras\keras_offline_win10\conda-pip-packages\conda> conda install vs2015_runtime-14.0.25420-0.tar.bz2
(venv) C:\Projects\anaconda_keras\keras_offline_win10\conda-pip-packages\conda> conda install setuptools-27.2.0-py35_1.tar.bz2
(venv) C:\Projects\anaconda_keras\keras_offline_win10\conda-pip-packages\conda> conda install scipy-0.19.1-np113py35_0.tar.bz2
```

conda 폴더로 이동하여 필요한 conda 패키지를 설치합니다.

```
(venv) C:\Projects\anaconda_keras\keras_offline_win10\conda-pip-packages\conda> cd ..\pip
(venv) C:\Projects\anaconda_keras\keras_offline_win10\conda-pip-packages\pip> pip install --no-index --find-links=. tensorflow-gpu    (GPU 사용 TensorFlow)
(venv) C:\Projects\anaconda_keras\keras_offline_win10\conda-pip-packages\pip> pip install --no-index --find-links=. tensorflow    (CPU 사용 TensorFlow)
(venv) C:\Projects\anaconda_keras\keras_offline_win10\conda-pip-packages\pip> pip install --no-index --find-links=. keras
```

pip 폴더로 이동하여 TensorFlow와 Keras pip 패키지를 설치합니다. TensorFlow는 원하는 아키텍쳐에 따라 tensorflow-gpu 또는 tensorflow 둘 중 하나만 설치하시면 됩니다.

### 설치 테스트

Python 인터프리터를 실행하여 Keras를 임포트해 봅니다. Using TensorFlow backend. 라고 출력되면 성공입니다.

```
(venv) C:\Projects\anaconda_keras\keras_offline_win10\conda-pip-packages\pip> python
>>> import keras
Using TensorFlow backend.
```

![testkeras]({{ "/images/2017-09-04-python.png" | prepend: site.baseurl }})

(2017/09/19 추가) 설치 후에 import keras를 할때 NumPy 관련 오류가 나는 경우가 있습니다. 그런 경우 pip 폴더에서 다음 명령어를 입력해 NumPy를 재설치 해 보시기 바랍니다. (성선경님 감사합니다)

```
pip install numpy-1.13.1-cp35-none-win_amd64.whl
```