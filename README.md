<div align="center">

# Cream🍦: Visually-Situated Natural Language Understanding with Contrastive Reading Model and Frozen Large Language Models
[![Paper](https://img.shields.io/badge/Paper-arxiv.2305.15080-orange)](https://arxiv.org/abs/2305.15080)
[![Conference](https://img.shields.io/badge/EMNLP-2023-red)](#how-to-cite)

Official Implementation of Cream | [Paper](https://arxiv.org/abs/2305.15080) | [Data](https://huggingface.co/naver-clova-ix) | [Slide](https://docs.google.com/presentation/d/1KziORzEtfHelFmhDWGq9WXIlQzTSGHoiOylxbSK4jPs/edit?usp=sharing) | [Poster](https://drive.google.com/file/d/1nYzBgy9a1o0BunD5jsywPecGpPJakCz1/view?usp=sharing)
</div>

## Introduction
Cream (Contrastive Reading Model) is a language-image understanding module designed to enhance the visually-situated natural language understanding capability in Large Language Models (LLMs). The primary goal of Cream is to effectively capture intricate details in images (e.g., texts), ensuring accurate responses in various visual langauge precessing applications.

Our academic paper, which describes our method in detail and provides full experimental results and analyses, can be found here:<br>
> [**Visually-Situated Natural Language Understanding with Contrastive Reading Model and Frozen Large Language Models**](https://arxiv.org/abs/2305.15080).<br>
> [Geewook Kim](https://geewook.kim), [Hodong Lee](https://scholar.google.com/citations?user=XRuGyvkAAAAJ), [Daehee Kim](https://scholar.google.com/citations?user=x_tWgpsAAAAJ), [Haeji Jung](https://scholar.google.com/citations?user=cyhTZ0MAAAAJ), [Sanghee Park](https://scholar.google.com/citations?user=_ryVHp0AAAAJ), [Yoonsik Kim](https://scholar.google.com/citations?user=nuxd_BsAAAAJ), [Sangdoo Yun](https://sangdooyun.github.io), [Taeho Kil](https://scholar.google.co.kr/citations?user=cV4h5MsAAAAJ), [Bado Lee](https://scholar.google.co.kr/citations?user=UAcfGOgAAAAJ), [Seunghyun Park](https://scholar.google.com/citations?user=iowjmTwAAAAJ). In EMNLP 2023.


## Updates

***2024-01-16*** Fix minor errors/typos. Release the PyPi package (`pip install cream-python`). Further updates will follow shortly.

***2023-12-06*** First commit with a codebase.

## Software Installation

```bash
pip install cream-python
```

or clone this repository and install the dependencies:

```bash
git clone https://github.com/naver-ai/cream.git
cd cream/
conda create -n cream_official python=3.8
conda activate cream_official
pip install .
```

If you want to run `train.py` or `test.py`, please also install other dependencies:

```bash
pip install -r requirements.txt
```


## Citation

If you find our work useful in your work, please consider citing our paper:

```
@inproceedings{kim2023cream,
      title={Visually-Situated Natural Language Understanding with Contrastive Reading Model and Frozen Large Language Models}, 
      author={Geewook Kim and Hodong Lee and Daehee Kim and Haeji Jung and Sanghee Park and Yoonsik Kim and Sangdoo Yun and Taeho Kil and Bado Lee and Seunghyun Park},
      booktitle = {Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing},
      year={2023},
}
```

## License

```
MIT license

Copyright (c) 2023-present NAVER Cloud Corp.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
```
