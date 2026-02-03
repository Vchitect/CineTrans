<div align="center">
  
<h1>CineTrans: Learning to Generate Videos with Cinematic Transitions via Masked Diffusion Models</h1>

[![](https://img.shields.io/static/v1?label=CineTrans&message=Project&color=purple)](https://uknowsth.github.io/CineTrans/)   [![](https://img.shields.io/static/v1?label=Paper&message=Arxiv&color=red&logo=arxiv)](https://arxiv.org/abs/2508.11484)   [![](https://img.shields.io/static/v1?label=Code&message=Github&color=blue&logo=github)](https://github.com/Vchitect/CineTrans)   [![](https://img.shields.io/static/v1?label=Dataset&message=HuggingFace&color=yellow&logo=huggingface)](https://huggingface.co/datasets/NumlockUknowSth/Cine250K)   

                    
<p><a href="https://scholar.google.com/citations?hl=zh-CN&user=TbZZSVgAAAAJ">Xiaoxue Wu</a><sup>1,2*</sup>,
<a href="https://scholar.google.com/citations?user=0gY2o7MAAAAJ&amp;hl=zh-CN" target="_blank">Bingjie Gao</a><sup>2,3</sup>,
<a href="https://scholar.google.com.hk/citations?user=gFtI-8QAAAAJ&amp;hl=zh-CN">Yu Qiao</a><sup>2&dagger;</sup>,
<a href="https://wyhsirius.github.io/">Yaohui Wang</a><sup>2&dagger;</sup>,
<a href="https://scholar.google.com/citations?user=3fWSC8YAAAAJ">Xinyuan Chen</a><sup>2&dagger;</sup></p>


<span class="author-block"><sup>1</sup>Fudan University</span>
<span class="author-block"><sup>2</sup>Shanghai Artificial Intelligence Laboratory</span>
<span class="author-block"><sup>3</sup>Shanghai Jiao Tong University</span>


<span class="author-block"><sup>*</sup>Work done during internship at Shanghai AI Laboratory</span> <span class="author-block"><sup>&dagger;</sup>Corresponding author</span>

</div>

## 🎥 Demo
https://github.com/user-attachments/assets/6f112e2f-40e3-4347-bab8-3e08bfa9366c

## 🔥 Updates
- [x] Release [Cine250K Dataset](https://huggingface.co/datasets/NumlockUknowSth/Cine250K) (26.2)
- 🎉🎉🎉 Our work has been accepted to ICLR 2026!
- [x] Release [model checkpoints](https://huggingface.co/NumlockUknowSth/CineTrans-DiT)
- [x] Release inference code
- [x] Release [arXiv paper](https://arxiv.org/pdf/2508.11484) 
- [x] Release [project page](https://uknowsth.github.io/CineTrans/)

## 📥 Installation
1. Clone the Repository
```
git clone https://github.com/UknowSth/CineTrans.git
cd CineTrans
```
2. Set up Environment
```
conda create -n cinetrans python==3.11.9
conda activate cinetrans

pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
```

## 🤗 Checkpoint  
### CineTrans-Unet
Download the required [model weights](https://huggingface.co/NumlockUknowSth/CineTrans-Unet/tree/main) and place them in the `ckpt/` directory.
```
ckpt/
│── stable-diffusion-v1-4/
│   ├── scheduler/
│   ├── text_encoder/
│   ├── tokenizer/  
│   │── unet/
│   └── vae_temporal_decoder/
│── checkpoint.pt
│── longclip-L.pt
```
### CineTrans-DiT
Download the weights of [Wan2.1-T2V-1.3B](https://huggingface.co/Wan-AI/Wan2.1-T2V-1.3B/tree/main) and [lora weights](https://huggingface.co/NumlockUknowSth/CineTrans-DiT/tree/main). Place them as:
```
Wan2.1-T2V-1.3B/ # original weights
│── google/
│   └── umt5-xxl/
│── config.json
│── diffusion_pytorch_model.safetensors
│── models_t5_umt5-xxl-enc-bf16.pth
│── Wan2.1_VAE.pth
ckpt/
└── weights.pt # lora weights
```
## 🖥️ Inference  
To run the inference, use the following command:
### CineTrans-Unet
```
python pipelines/sample.py --config configs/sample.yaml
```
Using a single A100 GPU, generating a single video takes approximately 40s. You can modify the relevant configurations and prompt in `configs/sample.yaml` to adjust the generation process.
### CineTrans-DiT
```
python generate.py
```
Using a single A100 GPU, generating a single video takes approximately 5min. You can modify the relevant configurations and prompt in `configs/t2v.yaml` to adjust the generation process.
## 🖼️ Gallery  

| ![vintage](https://github.com/user-attachments/assets/96aa859f-e8cc-4efd-802d-417cfafcf764) | ![city_night](https://github.com/user-attachments/assets/d9e3644c-1bb3-43c6-a1dd-ea4f816c04f2) | ![sea](https://github.com/user-attachments/assets/2f80ddac-d339-4e1d-83f4-962489e2a464) |
| ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ |
| Shot1:[0s,2.5s] Shot2:[2.5s,5s] Shot3:[5s,8s] | Shot1:[0s,2.5s] Shot2:[2.5s,5s] Shot3:[5s,8s] | Shot1:[0s,3s] Shot2:[3s,6s] Shot3:[6s,8s] |
| ![man](https://github.com/user-attachments/assets/1411eaed-8845-46a4-832f-74cd02ce82ff) | ![CASE1_1](https://github.com/user-attachments/assets/8d366979-8098-4c6e-94e5-3c44815f6d20) |![CASE31](https://github.com/user-attachments/assets/a8336c03-1ea2-4b7b-b591-dd63873921bb) |
| Shot1:[0s,2.75s] Shot2:[2.75s,5s] | Shot1:[0s,2.75s] Shot2:[2.75s,5s] | Shot1:[0s,1s] Shot2:[1s,2.5s] Shot3:[2.5s,3.75s] Shot4:[3.75s,5s] |

## 📑 BiTeX  
If you find [CineTrans](https://github.com/Vchitect/CineTrans.git) useful for your research and applications, please cite using this BibTeX:
```
@misc{wu2025cinetranslearninggeneratevideos,
      title={CineTrans: Learning to Generate Videos with Cinematic Transitions via Masked Diffusion Models}, 
      author={Xiaoxue Wu and Bingjie Gao and Yu Qiao and Yaohui Wang and Xinyuan Chen},
      year={2025},
      eprint={2508.11484},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2508.11484}, 
}
```





