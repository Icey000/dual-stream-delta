# SoccerNet Dual-Stream Dense Video Captioning

## Overview
A Dense Video Captioning (DVC) and Action Spotting system for the SoccerNet dataset. This project was originally built upon the Deltalab baseline (https://github.com/gladuz/soccernet-caption-deltalab) and serves as an **Architectural Exploration** and **Proof of Concept** for multi-modal feature fusion in sports video comprehension. The primary objective is to investigate how late-fusion mechanisms can integrate distinct streams of information to contextualize complex broadcasts.

## Core Innovation
The core innovation of this repository is the refactoring of the original single-stream visual Q-Former into a **Dual-Stream Q-Former**. By introducing an independent Audio stream alongside the Video stream, we implemented a multimodal audio-visual fusion mechanism directly before the downstream task modules. This decoupled approach allows both modalities to be processed asynchronously before being effectively combined.

## Architecture

![Dual-Stream Architecture](./assets/arch.png)

The Dual-Stream Video+Audio Q-Former fusion logic is designed to dynamically align and synthesize multimodal broadcast data. Individual visual and audio abstractions (such as Baidu/ResNET and LAION-CLAP embeddings) are parsed into their respective encoders. Subsequently, a late-fusion Q-Former layer actively queries both the dense visual representations and the audio temporal representations. By employing learned cross-attention mechanisms without forcibly reshaping disparate latent spaces, the Q-Former dynamically attends to synchronous audio cues (e.g., whistle blow, crowd cheer) and visual cues (e.g., player kicks, penalty cards), fusing them into an enriched, unified representation before passing the embeddings to the autoregressive decoders.

## Key Features
- **Dual-Stream Integration**: Seamless parallel processing of combined visual sequence tokens and CLAP audio tokens during forward passes.
- **Robust Audio Pipeline**: A comprehensive suite of diagnostic scripts designed to gracefully download, extract, and handle corrupted audio tracks from raw SoccerNet MKV sources.
- **Adaptive Fallback Mechanisms**: Built-in dummy audio generation (`4_generate_dummy_audio.py`) allows inference to proceed seamlessly relying strictly on the vision stream when raw audio is irretrievably damaged.
- **Unified Training Loop**: Multi-task learning capability supporting Dense Video Captioning, Action Spotting, and Classification through centralized batch unpacking.

## Installation

### Prerequisites
It is highly recommended to manage the environment utilizing `conda` or `mamba`.

```bash
# Create and activate environment
conda create -n soccernet-dvc python=3.9
conda activate soccernet-dvc

# Install PyTorch (Modify for your specific CUDA version)
pip install torch torchvision torchaudio

# Install Audio and general dependencies
pip install librosa numpy pandas
pip install git+https://github.com/LAION-AI/CLAP.git

# Install the SoccerNet core toolkit
pip install SoccerNet
```

## Quick Start

### 1. Data Preparation
Follow the numeric order of the preparation scripts to download the dataset and extract Audio features. 

```bash
# Configure your paths as shell variables
export SOCCERNET_VISION="./data/caption-2024/"
export SOCCERNET_AUDIO="./data/SoccerNet"
export SOCCERNET_ROOT=$SOCCERNET_AUDIO

# 1. Download MKVs and Extract WAV files
python 1_download_and_extract.py --quality 720p

# 2. Check for damaged audio and repair
python 3_check_wav_integrity.py 

# 3. Extract CLAP Embeddings
python 2_extract_clap_features.py 

# 4. Generate padding for natively silent videos
python 4_generate_dummy_audio.py
```

### 2. Training
Use the top-level scripts to initiate training and inference. You can modify hyperparameters directly in the bash script.
```bash
bash run_train.sh
```

### 3. Inference
To evaluate the model, run the testing script format.
```bash
bash run_test.sh
```

## Acknowledgments
This research and open-source release were heavily inspired by the pioneering work of the **SoccerNet** community and the **Deltalab** dense video captioning baseline. We thank the original authors for opening their source code, which facilitated this architectural exploration.
