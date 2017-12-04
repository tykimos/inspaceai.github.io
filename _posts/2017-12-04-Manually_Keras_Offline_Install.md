---
layout: post
title: "텐서플로우, 케라스 오프라인 설치 바이너리 생성법"
author: "정한솔"
date: 2017-12-04 14:00:00
categories: Lecture
comments: true
---

[Windows 7]({{ "/lecture/2017/10/13/Windows_7_Keras_Offline_Install.html" | prepend: site.baseurl }}) 또는 [Windows 10에서 텐서플로우, 케라스 오프라인 설치]({{ "/lecture/2017/10/13/Windows_10_Keras_Offline_Install.html" | prepend: site.baseurl }}) 포스트에서 Windows에서 텐서플로우, 케라스를 오프라인 설치하는 방법에 대해서 이미 설명하고 있습니다.

이 포스트에서는 해당 포스트에서 배포해드리고 있는 설치 파일을 직접 다운로드 받는 방법에 대해 설명드리겠습니다. 제가 보내드리는 패키지보다 최신 버전의 라이브러리를 사용하고 싶거나, 패키지의 구성을 바꾸고 싶거나, 제가 메일을 보내드리는 시간 없이 바로 오프라인 설치 파일이 필요할 경우 이 방법을 사용하시기 바랍니다.

다운로드를 위해 오프라인 설치를 원하시는 환경과 같은 운영체제가 설치되고 인터넷에 연결되어 있는 다운로드용 환경을 준비합니다. 아래의 작업은 모두 다운로드용 환경에서 시행하시면 되며 USB 메모리 등 오프라인 저장 매체를 사용하여 오프라인 환경으로 옮기신 다음 위 포스트의 설치 방법을 따라 설치하시면 됩니다. 단 제가 포스트를 작성했을 때와 비교하여 패키지들의 버전이 업데이트되어 있을 수 있고 따라서 패키지 파일 명이 달라질 수 있음을 유의하시기 바랍니다.

---

### CUDA

nVIDIA GPU에서 CUDA를 통해 텐서플로우를 가속하고자 하시는 경우 CUDA 툴킷과 cuDNN 라이브러리가 필요합니다. 현재 CUDA의 최신 버전은 9.0이지만, [현재 텐서플로우에서는 8.0 버전만을 지원](https://www.tensorflow.org/install/install_windows)하므로 구 버전을 찾아서 다운로드 받으셔야 합니다. CUDA 8.0 다운로드는 [이곳](https://developer.nvidia.com/cuda-80-ga2-download-archive)에서 다운로드 받으실 수 있습니다. 원하시는 운영체제를 선택하시고 인스톨러를 다운로드 받되 Installer Type은 반드시 exe (local)을 선택하시기 바랍니다.

cuDNN은 [이곳](https://developer.nvidia.com/cudnn)에서 다운로드 받으실 수 있는데 NVIDIA 개발자 계정으로 로그인을 할 필요가 있습니다. 가입은 무료이며 이메일과 간략한 설문조사 외에 요구받는 것은 없습니다. cuDNN의 최신 버전은 7.0인데 역시나 텐서플로우가 지원하지 않으므로 6.0 버전을 설치하셔야 합니다. CUDA 8.0을 위한 cuDNN 6.0 라이브러리를 찾아 자신에게 맞는 운영체제를 선택하여 다운로드 받습니다.

---

### Anaconda

[이곳](https://conda.io/miniconda.html)에서 미니콘다를 다운로드 받아 설치합니다. 미니콘다는 아나콘다에 함께 포함된 많은 각종 수학, 과학 계산 패키지들을 제외하고 오직 아나콘다 본체만 설치할 수 있도록 한 버전입니다. 예전에 작성된 텐서플로우 설치 가이드 중에서는 Windows에서 텐서플로우가 Python 3.6을 지원하지 않는다는 말이 있는데 현재는 지원하므로 최신 3.6 버전을 다운로드 받아도 괜찮습니다.

아나콘다는 오프라인을 통한 패키지 설치 기능 지원이 빈약한 편입니다. 오프라인 패키지를 별도로 다운로드 받는 기능이 없어 한번 설치한 다음에 캐시 디렉토리에서 꺼내와야 하며, 로컬에서 패키지 의존을 추적하는 기능이 불완전해 모든 패키지를 하나 하나 수동으로 설치해야 합니다. 다행히도 [conda/constructor](https://github.com/conda/constructor)라는 툴에서 아나콘다 본체와 지정된 아나콘다 패키지와 그 의존 패키지를 모두 묶어서 하나의 인스톨러로 내보내는 기능을 제공합니다. 이 툴을 사용하면 오프라인 환경에서 아나콘다 환경을 한번에 구축하는 인스톨러를 생성할 수 있습니다.

시작 메뉴에서 Anaconda Prompt를 검색하여 실행한 다음에 다음 명령어를 차례대로 입력합니다.

```
conda create --name constructor
activate constructor
conda install constructor
```

그리고 빈 작업 폴더를 하나 만들고 그 작업 폴더 안으로 이동합니다. 그 작업 폴더 아래에 construct.yaml 파일을 하나 생성합니다. 이 파일은 인스톨러에 포함될 아나콘다 패키지에 관련된 정보를 담고 있어야 합니다. 제가 사용한 construct.yaml 파일의 내용은 아래와 같습니다.

```yaml
name: InspaceDllab
version: 1.1.0

channels:
  - http://repo.continuum.io/pkgs/free/

specs:
  - python
  - conda
  - pip
  - pandas
  - matplotlib
  - jupyter
  - notebook
  - scikit-learn
```

추가로 필요한 패키지가 있으면 specs 아래에 추가할 수 있습니다. 작성한 construct.yaml 파일을 저장한 후 콘솔창에 다음 명령어를 입력하면 인스톨러가 생성됩니다. 다운로드 및 인스톨러 생성에 약간 시간이 걸릴 수 있습니다.

```
constructor .
```

---

### pip

pip는 오프라인 설치를 위해 패키지와 관련 의존을 자동으로 다운로드 받는 기능을 제공하기 때문에 과정이 간단합니다. 우선 미니콘다 초기 상태에서는 pip조차 설치되어 있지 않기 때문에 pip 를 먼저 설치하고 난 다음에 pip download 명령을 통해 필요 패키지를 다운로드 받으실 수 있습니다.

```
conda install pip
pip download tensorflow tensorflow-gpu keras opencv-python h5py pydot pillow
```

pip를 통한 다운로드가 잘 되지 않거나 좀더 최적화된 바이너리가 필요한 경우 [UCI](https://uci.edu/) [LFD](https://www.lfd.uci.edu/)에서 제공하는 [비공식 Python 패키지 바이너리](https://www.lfd.uci.edu/~gohlke/pythonlibs/)를 참고하시면 도움이 될 수 있습니다.
