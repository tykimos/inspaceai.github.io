---
layout: post
title: "텐서플로우, 케라스 오프라인 설치 바이너리 생성법"
author: "정한솔"
date: 2017-12-04 14:00:00
categories: Lecture
comments: true
---

이 포스트에서는 제가 배포해드리고 있는 설치 파일을 직접 다운로드 받는 방법에 대해 설명드리겠습니다. 제가 보내드리는 패키지보다 최신 버전의 라이브러리를 사용하고 싶거나, 패키지의 구성을 바꾸고 싶거나, 제가 메일을 보내드리는 시간 없이 바로 오프라인 설치 파일이 필요할 경우 이 방법을 사용하시기 바랍니다.

[Windows 7]({{ "/lecture/2017/10/13/Windows_7_Keras_Offline_Install.html" | prepend: site.baseurl }}) 또는 [Windows 10에서 텐서플로우, 케라스 오프라인 설치]({{ "/lecture/2017/10/13/Windows_10_Keras_Offline_Install.html" | prepend: site.baseurl }}) 포스트에서 Windows에서 텐서플로우, 케라스를 오프라인 설치하는 방법에 대해서 이미 설명하고 있습니다.

다운로드를 위해 오프라인 설치를 원하시는 환경과 같은 운영체제가 설치되고 인터넷에 연결되어 있는 다운로드용 환경을 준비합니다. 아래의 작업은 모두 다운로드용 환경에서 시행하시면 되며 USB 메모리 등 오프라인 저장 매체를 사용하여 오프라인 환경으로 옮기신 다음 위 포스트의 설치 방법을 따라 설치하시면 됩니다. 단 제가 포스트를 작성했을 때와 비교하여 패키지들의 버전이 업데이트되어 있을 수 있고 따라서 패키지 파일 명이 달라질 수 있음을 유의하시기 바랍니다.

 * 2018/2/20 변경된 설치 방법 및 업데이트된 최신 패키지 버전에 맞춰 포스트 내용을 수정하였습니다.

---

### CUDA

nVIDIA GPU에서 CUDA를 통해 텐서플로우를 가속하고자 하시는 경우 CUDA 툴킷과 cuDNN 라이브러리가 필요합니다. 현재 CUDA의 최신 버전은 9.1이지만, [현재 텐서플로우 바이너리에서는 9.0을 사용](https://github.com/tensorflow/tensorflow/releases/tag/v1.5.0)하므로 구 버전을 찾아서 다운로드 받으셔야 합니다. CUDA 9.0 다운로드는 [이곳](https://developer.nvidia.com/cuda-90-download-archive)에서 다운로드 받으실 수 있습니다. 원하시는 운영체제를 선택하시고 인스톨러를 다운로드 받되 Installer Type은 반드시 exe (local)을 선택하시기 바랍니다.

cuDNN은 [이곳](https://developer.nvidia.com/cudnn)에서 다운로드 받으실 수 있는데 NVIDIA 개발자 계정으로 로그인을 할 필요가 있습니다. 가입은 무료이며 이메일과 간략한 설문조사 외에 요구받는 것은 없습니다. cuDNN의 최신 버전은 7.0이므로 CUDA 9.0을 위한 cuDNN 7.0 라이브러리를 찾아 자신에게 맞는 운영체제를 선택하여 다운로드 받습니다.

---

### Python

파이썬은 풍부한 계산 라이브러리를 가지고 있어 딥 러닝, 머신 러닝 관련 알고리즘을 개발 또는 구현하는데 널리 사용되고 있습니다. 텐서플로우 및 케라스도 파이썬으로 작성된 라이브러리입니다.

[이곳](https://www.python.org/downloads/windows/)에서 파이썬 최신 버전을 다운로드 받으실 수 있습니다. 반드시 x86-64 excutable installer를 선택하셔야 합니다. 아래의 pip 패키지를 다운로드 받기 위해 다운로드용 환경에도 파이썬을 설치하셔야 합니다. 설치 과정 중 Add Python 3.6 to PATH의 체크 박스는 체크를 하고 진행하시기 바랍니다.

---

### pip

시작 메뉴에서 `cmd`를 입력하여 명령 프롬프트를 실행한 뒤 다음 명령어를 실행하여 패키지를 다운로드 받으실 수 있습니다.

```
python -m pip download tensorflow tensorflow-gpu keras jupyter opencv-python h5py pydot pillow matplotlib
```

pip에는 인코딩 관련 버그가 있어 위 다운로드 명령이 잘 실행되지 않을 수 있습니다. 이 경우 최신 베타 버전을 설치해서 재시도 해 보시기 바랍니다. [git](https://git-scm.com/)을 시스템에 설치하고 다음 명령어를 사용하시면 됩니다.

```
python -m pip install --upgrade git+https://github.com/pypa/pip.git`
```

matplotlib를 사용하기 위해서는 시스템에 Visual C++ 2010 런타임 라이브러리를 설치할 필요성이 있습니다. [이곳](https://www.microsoft.com/en-us/download/details.aspx?id=14632)에서 다운로드 받으시면 됩니다.
