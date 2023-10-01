<div align= "center">
    <h1>🤖 ConvRe 🤯</h1>
</div>

<div align="center">

![Dialogues](https://img.shields.io/badge/Relation\_Num-17-red?style=flat-square)
![Dialogues](https://img.shields.io/badge/Triple\_num-1240-yellow?style=flat-square)
![Dialogues](https://img.shields.io/badge/Version-1.0-green?style=flat-square)

</div>


<p align="center">
  <a href="#data">Data Release</a> •
  <a href="#web-ui">Huggingface Leaderboard</a> •
  <a href="assets/paper.pdf">Paper</a> •
  <a href="#citation">Citation</a>

</p>



🤖🤯This project (ConvRe) aims to evaluate how good LLMs handle converse relations that are less common during training. XX. 


*Read this in [中文](README_ZH.md).*

## What's New
- **[2023/10/08]** **ConvRe** benchmark is released.


## 🥝Data

ConvRe is XX.

You can download it from XX

## 🤖Supported Models
The models listed below are supported and can be run using the script in Inference.

## Inference with huggingface dataset
We provide a convenient way to run the experiments and there are only three parameters that need to consider:
- `model_name`: the name of the large language model you want to use.
- `task`: the subtasks of ConvRe benchmark: text2re or re2text.
- `settings`: prompt setting for current run (prompt1 to prompt 12), please refer to our paper(LINK) for more details of each setting.

**Example**

If you want to run `prompt1` of `re2text` task on `flan-t5-base`, run this script
```bash
python3 main_hf.py --model_name flan-t5-base --task re2text --settings prompt1
```

## Inference using local dataset 

The parameter setting for each prompt is listed below

| Prompt ID  |  prompt  | relation | n_shot | example_type | text_type |
|:----------:|:--------:|:--------:|:------:|:------------:|:---------:|
| re2text 1# |  normal  |  normal  |   0    |   regular    |  regular  |
| text2re 1# |  normal  |  normal  |   0    |   regular    |   hard    |
| re2text 2# |  normal  |  normal  |   0    |   regular    |   hard    |
| text2re 2# |  normal  |  normal  |   0    |   regular    |  regular  |
| re2text 3# |  normal  | converse |   0    |   regular    |  regular  |
| text2re 3# |  normal  | converse |   0    |   regular    |   hard    |
| re2text 4# |  normal  | converse |   0    |   regular    |   hard    |
| text2re 4# |  normal  | converse |   0    |   regular    |  regular  |
| re2text 5# |   hint   | converse |   0    |   regular    |  regular  |
| text2re 5# |   hint   | converse |   0    |   regular    |   hard    |
| re2text 6# |   hint   | converse |   0    |   regular    |   hard    |
| text2re 6# |   hint   | converse |   0    |   regular    |  regular  |
|     7#     |  normal  | converse |   3    |     hard     |   hard    |
|     8#     | hint+cot | converse |   3    |     hard     |   hard    |
|     9#     |  normal  | converse |   6    |     hard     |   hard    |
|    10#     |  normal  | converse |   3    |   regular    |   hard    |
|    11#     | hint+cot | converse |   3    |   regular    |   hard    |
|    12#     |  normal  | converse |   6    |   regular    |   hard    |

## Citation