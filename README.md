# Neural Networks: Theory and Implementation - Term Project

This repository contains the implementation of the term project for the course **Neural Networks: Theory and Implementation**. It was implemented by [**Dhimitrios Duka**](dhimitrios.duka1@gmail.com) and [**Kai Wittenmayer**](kawi00002@uni-saarland.de). The project focuses on fine-tuning multilingual language models (LLMs) using parameter-efficient tuning (PEFT) methods like **BitFit**, **LoRA**, and **IA3**. The project evaluates the performance of these techniques on underrepresented languages, particularly **quy_Latn** (Quechua).

## Table of Contents
- [Project Overview](#project-overview)
- [Datasets](#datasets)
- [Results](#results)
- [References](#references)

## Project Overview
The goal of this project is to explore and compare various fine-tuning techniques for adapting multilingual language models to low-resource languages. Specifically, we aim to enhance the performance of models such as **XGLM-564M** and **GPT-2** in non-dominant languages, with a focus on **quy_Latn**. The fine-tuning methods analyzed include full fine-tuning and parameter-efficient approaches like **BitFit**, **LoRA**, and **IA3**.

## Datasets
The following datasets were used for training and evaluation:
- **NLLB** (No Language Left Behind): A dataset with diverse multilingual content.
- **Spanish-to-Quechua**: A dataset for translating Spanish to the underrepresented Quechua language.

Additional datasets such as **OSCAR** and **CC100** were evaluated but filtered due to quality and length concerns.

## Results
The performance of the models was measured using language modeling loss. Notably, full fine-tuning showed the best results with a **30.4% improvement** in loss for **quy_Latn**, while **BitFit**, **LoRA**, and **IA3** offered competitive results with far fewer parameter updates. Below is a comparison of losses:

| Method        | Loss on quy_Latn |
| ------------- |:----------------:|
| Full Fine-Tuning | 4.90 |
| BitFit         | 5.22 |
| LoRA           | 5.23 |
| IA3            | 5.22 |

The fine-tuned models were visualized using **PCA** and **t-SNE** to explore multilingual representation spaces.

## References
- [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685)
- [BitFit: Simple Parameter-Efficient Fine-Tuning](https://arxiv.org/abs/2106.10199)
- [IA3: Parameter-Efficient Tuning](https://arxiv.org/abs/2205.05638)
- [NLLB: No Language Left Behind](https://arxiv.org/abs/2207.04672)
