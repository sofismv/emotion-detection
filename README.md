# Emotion detection. NLP Course Project.
Information about method in this report  
Authors: Sofia Samoilova, Dmitrii Sutyi, Evgeniy Kalimullin  
About this task https://github.com/emotion-analysis-project/SemEval2025-Task11  
track A in rus:  
[Official Ranking](https://docs.google.com/spreadsheets/d/1pyKXpvmVsE1rwu8aRnfn372rD-v3L4Qo/edit?gid=669615273#gid=669615273)  
[Unofficial Ranking](https://docs.google.com/spreadsheets/d/1IgPfmL0z9Lc7GCVY3LwOsHEGQTvOjtZ0/edit?gid=348498964#gid=348498964)
## Paper With Results
[Emotion detection in russian language](https://github.com/sofismv/emotion-detection/blob/main/paper.pdf).

## Prerequisites
⋅⋅*10GB GPU memory(we use NVIDIA GeForce RTX 2080 Ti)


## Setup
1. Create environment
```bash
conda env create -f environment.yaml
```
```bash
conda activate emotion_detection
```
2. Check config/config.yaml and fix file 
3. Run train.py

## Repo structure
<pre><code>.
├── README.md
├── config
│   └── config.yaml
├── environment.yaml
├── notebooks
│   ├── few_shot_prompt.ipynb
│   ├── rubert.ipynb
│   └── zero_shot_prompt.ipynb
├── src
│   └── train.py
└── tree.txt
 </code> </pre>

## Report
https://api.wandb.ai/links/sofism/wjjjfh8l
