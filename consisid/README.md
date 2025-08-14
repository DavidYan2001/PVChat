<div align=center>
<img src="https://github.com/PKU-YuanGroup/ConsisID/blob/main/asserts/ConsisID_logo.png?raw=true" width="150px">
</div>
<h2 align="center"> <a href="https://arxiv.org/abs/2411.17440">Identity-Preserving Text-to-Video Generation by Frequency Decomposition</a></h2>

<h5 align="center"> If you like our project, please give us a star ⭐ on GitHub for the latest update.  </h2>


<h5 align="center">


[![hf_space](https://img.shields.io/badge/🤗-Open%20In%20Spaces-blue.svg)](https://huggingface.co/spaces/BestWishYsh/ConsisID-preview-Space)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/camenduru/ConsisID-jupyter/blob/main/ConsisID_jupyter.ipynb)
[![hf_paper](https://img.shields.io/badge/🤗-Paper%20In%20HF-red.svg)](https://huggingface.co/papers/2411.17440)
[![arXiv](https://img.shields.io/badge/Arxiv-2411.17440-b31b1b.svg?logo=arXiv)](https://arxiv.org/abs/2411.17440) 
[![Home Page](https://img.shields.io/badge/Project-<Website>-blue.svg)](https://pku-yuangroup.github.io/ConsisID/) 
[![Dataset](https://img.shields.io/badge/Dataset-previewData-green)](https://huggingface.co/datasets/BestWishYsh/ConsisID-preview-Data)
[![zhihu](https://img.shields.io/badge/-Twitter@Adina%20Yakup%20-black?logo=twitter&logoColor=1D9BF0)](https://x.com/AdinaYakup/status/1862604191631573122)
[![zhihu](https://img.shields.io/badge/-Twitter@camenduru%20-black?logo=twitter&logoColor=1D9BF0)](https://x.com/camenduru/status/1861957812152078701)
[![zhihu](https://img.shields.io/badge/-YouTube-000000?logo=youtube&logoColor=FF0000)](https://www.youtube.com/watch?v=PhlgC-bI5SQ)
[![License](https://img.shields.io/badge/License-Apache%202.0-yellow)](https://github.com/PKU-YuanGroup/ConsisID/blob/main/LICENSE) 
[![github](https://img.shields.io/github/stars/PKU-YuanGroup/ConsisID.svg?style=social)](https://github.com/PKU-YuanGroup/ConsisID/)

</h5>

<div align="center">
This repository is the official implementation of ConsisID, a tuning-free DiT-based controllable IPT2V model to keep human-identity consistent in the generated video. The approach draws inspiration from previous studies on frequency analysis of vision/diffusion transformers.
</div>






<br>

<details open><summary>💡 We also have other video generation projects that may interest you ✨. </summary><p>
<!--  may -->

> [**Open-Sora Plan: Open-Source Large Video Generation Model**](https://arxiv.org/abs/2412.00131) <br>
> Bin Lin, Yunyang Ge and Xinhua Cheng etc. <br>
[![github](https://img.shields.io/badge/-Github-black?logo=github)](https://github.com/PKU-YuanGroup/Open-Sora-Plan)  [![github](https://img.shields.io/github/stars/PKU-YuanGroup/Open-Sora-Plan.svg?style=social)](https://github.com/PKU-YuanGroup/Open-Sora-Plan) [![arXiv](https://img.shields.io/badge/Arxiv-2412.00131-b31b1b.svg?logo=arXiv)](https://arxiv.org/abs/2412.00131) <br>
>
> [**MagicTime: Time-lapse Video Generation Models as Metamorphic Simulators**](https://arxiv.org/abs/2404.05014) <br>
> Shenghai Yuan, Jinfa Huang and Yujun Shi etc. <br>
> [![github](https://img.shields.io/badge/-Github-black?logo=github)](https://github.com/PKU-YuanGroup/MagicTime)  [![github](https://img.shields.io/github/stars/PKU-YuanGroup/MagicTime.svg?style=social)](https://github.com/PKU-YuanGroup/MagicTime) [![arXiv](https://img.shields.io/badge/Arxiv-2404.05014-b31b1b.svg?logo=arXiv)](https://arxiv.org/abs/2404.05014) <br>
>
> [**ChronoMagic-Bench: A Benchmark for Metamorphic Evaluation of Text-to-Time-lapse Video Generation**](https://arxiv.org/abs/2406.18522) <br>
> Shenghai Yuan, Jinfa Huang and Yongqi Xu etc. <br>
> [![github](https://img.shields.io/badge/-Github-black?logo=github)](https://github.com/PKU-YuanGroup/ChronoMagic-Bench/)  [![github](https://img.shields.io/github/stars/PKU-YuanGroup/ChronoMagic-Bench.svg?style=social)](https://github.com/PKU-YuanGroup/ChronoMagic-Bench/) [![arXiv](https://img.shields.io/badge/Arxiv-2406.18522-b31b1b.svg?logo=arXiv)](https://arxiv.org/abs/2406.18522) <br>
> </p></details>


## 📣 News

* ⏳⏳⏳ Release the full codes & datasets & weights.
* ⏳⏳⏳ Integrate into Diffusers.
* `[2024.12.09]`  🔥We release the [test set](https://huggingface.co/datasets/BestWishYsh/ConsisID-preview-Data/tree/main/eval) and [metric calculation code](https://github.com/PKU-YuanGroup/ConsisID/tree/main/eval) used in the paper, now your can measure the metrics on your own machine. Please refer to [this guide](https://github.com/PKU-YuanGroup/ConsisID/tree/main/eval) for more details.
* `[2024.12.08]`  🔥The code for <u>data preprocessing</u> is out, which is used to obtain the [training data](https://huggingface.co/datasets/BestWishYsh/ConsisID-preview-Data) required by ConsisID. Please refer to [this guide](https://github.com/PKU-YuanGroup/ConsisID/tree/main/data_preprocess) for more details.
* `[2024.12.04]`  Thanks [@shizi](https://www.bilibili.com/video/BV1v3iUY4EeQ/?vd_source=ae3f2652765c02e41cdd698b311989e3) for providing [🤗Windows-ConsisID](https://huggingface.co/pkuhexianyi/ConsisID-Windows/tree/main) and [🟣Windows-ConsisID](https://www.wisemodel.cn/models/PkuHexianyi/ConsisID-Windows/file), which make it easy to run ConsisID on Windows.
* `[2024.12.01]`  🔥 We provide full text prompts corresponding to all the videos on project page. Click [here](https://github.com/PKU-YuanGroup/ConsisID/blob/main/asserts/prompt.xlsx) to get and try the demo.
* `[2024.11.30]`  🔥 We have fixed the [huggingface demo](https://huggingface.co/spaces/BestWishYsh/ConsisID-preview-Space), welcome to try it.
* `[2024.11.29]`  🔥 The current codes and weights are our early versions, and the differences with the latest version in [arxiv](https://github.com/PKU-YuanGroup/ConsisID) can be viewed [here](https://github.com/PKU-YuanGroup/ConsisID/tree/main/util/on_going_module). And we will release the full codes in the next few days.
* `[2024.11.28]`  Thanks [@camenduru](https://twitter.com/camenduru) for providing [Jupyter Notebook](https://colab.research.google.com/github/camenduru/ConsisID-jupyter/blob/main/ConsisID_jupyter.ipynb) and [@Kijai](https://github.com/kijai) for providing ComfyUI Extension [ComfyUI-ConsisIDWrapper](https://github.com/kijai/ComfyUI-CogVideoXWrapper). If you find related work, please let us know.
* `[2024.11.27]`  🔥 Due to policy restrictions, we only open-source part of the dataset. You can download it by clicking [here](https://huggingface.co/datasets/BestWishYsh/ConsisID-preview-Data). And we will release the data processing codes in the next few days.
* `[2024.11.26]`  🔥 We release the arXiv paper for ConsisID, and you can click [here](https://arxiv.org/abs/2411.17440) to see more details.
* `[2024.11.22]`  🔥 **All codes & datasets** are coming soon! Stay tuned 👀!

## 😍 Gallery

Identity-Preserving Text-to-Video Generation.

[![Demo Video of ConsisID](https://github.com/user-attachments/assets/634248f6-1b54-4963-88d6-34fa7263750b)](https://www.youtube.com/watch?v=PhlgC-bI5SQ)
or you can click <a href="https://github.com/SHYuanBest/shyuanbest_media/raw/refs/heads/main/ConsisID/showcase_videos.mp4">here</a> to watch the video.

## 🤗 Demo

### Gradio Web UI

Highly recommend trying out our web demo by the following command, which incorporates all features currently supported by ConsisID. We also provide [online demo](https://huggingface.co/spaces/BestWishYsh/ConsisID-preview-Space) in Hugging Face Spaces.

```bash
python app.py
```

### CLI Inference

```bash
python infer.py --model_path BestWishYsh/ConsisID-preview
```

warning: It is worth noting that even if we use the same seed and prompt but we change a machine, the results will be different.

### Prompt Refiner

ConsisID has high requirements for prompt quality. You can use [GPT-4o](https://chatgpt.com/) to refine the input text prompt, an example is as follows (original prompt: "a man is playing guitar.")
```bash
a man is playing guitar.

Change the sentence above to something like this (add some facial changes, even if they are minor. Don't make the sentence too long): 

The video features a man standing next to an airplane, engaged in a conversation on his cell phone. he is wearing sunglasses and a black top, and he appears to be talking seriously. The airplane has a green stripe running along its side, and there is a large engine visible behind his. The man seems to be standing near the entrance of the airplane, possibly preparing to board or just having disembarked. The setting suggests that he might be at an airport or a private airfield. The overall atmosphere of the video is professional and focused, with the man's attire and the presence of the airplane indicating a business or travel context.
```

Some sample prompts are available [here](https://github.com/PKU-YuanGroup/ConsisID/blob/main/asserts/prompt.xlsx).

### GPU Memory Optimization

ConsisID requires about 44 GB of GPU memory to decode 49 frames (6 seconds of video at 8 FPS) with output resolution 720x480 (W x H), which makes it not possible to run on consumer GPUs or free-tier T4 Colab. The following memory optimizations could be used to reduce the memory footprint. For replication, you can refer to [this](https://gist.github.com/SHYuanBest/bc4207c36f454f9e969adbb50eaf8258) script.

| Feature (overlay the previous) | Max Memory Allocated | Max Memory Reserved |
| :----------------------------- | :------------------- | :------------------ |
| -                              | 37 GB                | 44 GB               |
| enable_model_cpu_offload       | 22 GB                | 25 GB               |
| enable_sequential_cpu_offload  | 16 GB                | 22 GB               |
| vae.enable_slicing             | 16 GB                | 22 GB               |
| vae.enable_tiling              | 5 GB                 | 7 GB                |

```bash
# turn on if you don't have multiple GPUs or enough GPU memory(such as H100)
pipe.enable_model_cpu_offload()
pipe.enable_sequential_cpu_offload()
pipe.vae.enable_slicing()
pipe.vae.enable_tiling()
```
warning: it will cost more time in inference and may also reduce the quality.

## ⚙️ Requirements and Installation

We recommend the requirements as follows.

### Environment

```bash
git clone --depth=1 https://github.com/PKU-YuanGroup/ConsisID.git
cd ConsisID
conda create -n consisid python=3.11.0
conda activate consisid
pip install -r requirements.txt
```

### Download ConsisID

The weights are available at [🤗HuggingFace](https://huggingface.co/BestWishYsh/ConsisID-preview) and [🟣WiseModel](https://wisemodel.cn/models/SHYuanBest/ConsisID-Preview/file), and will be automatically downloaded when runing `app.py` and `infer.py`, or you can download it with the following commands.

```bash
# way 1
# if you are in china mainland, run this first: export HF_ENDPOINT=https://hf-mirror.com
cd util
python download_weights.py

# way 2
# if you are in china mainland, run this first: export HF_ENDPOINT=https://hf-mirror.com
huggingface-cli download --repo-type model \
BestWishYsh/ConsisID-preview \
--local-dir ckpts

# way 3
git lfs install
git clone https://www.wisemodel.cn/SHYuanBest/ConsisID-Preview.git
```

Once ready, the weights will be organized in this format:

```
📦 ckpts/
├── 📂 data_process/
├── 📂 face_encoder/
├── 📂 scheduler/
├── 📂 text_encoder/
├── 📂 tokenizer/
├── 📂 transformer/
├── 📂 vae/
├── 📄 configuration.json
├── 📄 model_index.json
```

## 🗝️ Training

### Data preprocessing

Please refer to [this guide](https://github.com/PKU-YuanGroup/ConsisID/tree/main/data_preprocess) for how to obtain the [training data](https://huggingface.co/datasets/BestWishYsh/ConsisID-preview-Data) required by ConsisID. If you want to train a text to image and video generation model. You need to arrange all the dataset in this [format](https://github.com/PKU-YuanGroup/ConsisID/tree/main/asserts/demo_train_data/dataname):

```
📦 datasets/
├── 📂 captions/
│   ├── 📄 dataname_1.json
│   ├── 📄 dataname_2.json
├── 📂 dataname_1/
│   ├── 📂 refine_bbox_jsons/
│   ├── 📂 track_masks_data/
│   ├── 📂 videos/
├── 📂 dataname_2/
│   ├── 📂 refine_bbox_jsons/
│   ├── 📂 track_masks_data/
│   ├── 📂 videos/
├── ...
├── 📄 total_train_data.txt
```

### Video DiT training

First, setting hyperparameters:

- environment (e.g., cuda): [deepspeed_configs](https://github.com/PKU-YuanGroup/ConsisID/tree/main/util/deepspeed_configs)
- training arguments (e.g., batchsize): [train_single_rank.sh](https://github.com/PKU-YuanGroup/ConsisID/blob/main/train_single_rank.sh) or [train_multi_rank.sh](https://github.com/PKU-YuanGroup/ConsisID/blob/main/train_multi_rank.sh)

Then, we run the following bash to start training:

```bash
# For single rank
bash train_single_rank.sh
# For multi rank
bash train_multi_rank.sh
```

## 🙌 Friendly Links

We found some plugins created by community developers. Thanks for their efforts: 

  - ComfyUI Extension. [ComfyUI-ConsisIDWrapper](https://github.com/kijai/ComfyUI-CogVideoXWrapper) (by [@Kijai](https://github.com/kijai)).
  - Jupyter Notebook. [Jupyter-ConsisID](https://colab.research.google.com/github/camenduru/ConsisID-jupyter/blob/main/ConsisID_jupyter.ipynb) (by [@camenduru](https://github.com/camenduru/consisid-tost)).
  - Windows Docker. [🤗Windows-ConsisID](https://huggingface.co/pkuhexianyi/ConsisID-Windows/tree/main) and [🟣Windows-ConsisID](https://www.wisemodel.cn/models/PkuHexianyi/ConsisID-Windows/file) (by [@shizi](https://www.bilibili.com/video/BV1v3iUY4EeQ/?vd_source=ae3f2652765c02e41cdd698b311989e3)).
  - Diffusers. We need your help to integrate ConsisID into [Diffusers](https://huggingface.co/docs/diffusers). 🙏 **[Need your contribution]**

If you find related work, please let us know. 

## 🐳 Dataset

We release the subset of the data used to train ConsisID. The dataset is available at [HuggingFace](https://huggingface.co/datasets/BestWishYsh/ConsisID-preview-Data), or you can download it with the following command. Some samples can be found on our [Project Page](https://pku-yuangroup.github.io/ConsisID/).

```bash
huggingface-cli download --repo-type dataset \
BestWishYsh/ConsisID-preview-Data \
--local-dir BestWishYsh/ConsisID-preview-Data
```

## 🛠️ Evaluation 

We release the data used for evaluation in [ConsisID](https://huggingface.co/papers/2411.17440), which is available at [HuggingFace](https://huggingface.co/datasets/BestWishYsh/ConsisID-preview-Data). Please refer to [this guide](https://github.com/PKU-YuanGroup/ConsisID/tree/main/eval) for how to evaluate customized model.

## 👍 Acknowledgement

* This project wouldn't be possible without the following open-sourced repositories: [Open-Sora Plan](https://github.com/PKU-YuanGroup/Open-Sora-Plan), [CogVideoX](https://github.com/THUDM/CogVideo), [EasyAnimate](https://github.com/aigc-apps/EasyAnimate), [CogVideoX-Fun](https://github.com/aigc-apps/CogVideoX-Fun).

## 🔒 License

* The majority of this project is released under the Apache 2.0 license as found in the [LICENSE](https://github.com/PKU-YuanGroup/ConsisID/blob/main/LICENSE) file.
* The CogVideoX-5B model (Transformers module) is released under the [CogVideoX LICENSE](https://huggingface.co/THUDM/CogVideoX-5b/blob/main/LICENSE).
* The service is a research preview. Please contact us if you find any potential violations. (shyuan-cs@hotmail.com)

## ✏️ Citation

If you find our paper and codes useful in your research, please consider giving a star :star: and citation :pencil:.

```BibTeX
@article{yuan2024identity,
  title={Identity-Preserving Text-to-Video Generation by Frequency Decomposition},
  author={Yuan, Shenghai and Huang, Jinfa and He, Xianyi and Ge, Yunyuan and Shi, Yujun and Chen, Liuhan and Luo, Jiebo and Yuan, Li},
  journal={arXiv preprint arXiv:2411.17440},
  year={2024}
}
```

## 🤝 Contributors

<a href="https://github.com/PKU-YuanGroup/ConsisID/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=PKU-YuanGroup/ConsisID&anon=true" />

</a>
