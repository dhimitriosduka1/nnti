# Language model adaptation

We saw in `Task1` that the language modeling loss of `XGLM` is much better on some languages than others. In this task, your goal is to improve the language modeling performance of `XGLM` on Ayacucho Quechua (`quy_Latn` in the flores dataset).

1. First, we will need to find suitable data to train our model. Explore the [huggingface datasets](https://huggingface.co/datasets) library for datasets that contain text in Quechua. It's up to you to decide how much and which data you are going to use for Task 3. The only rule is: **you are not allowed to use the flores data for training**. Try to be creative and investigate the choice of adaptation data and its quality for the remaining exercises.

2. To adapt our model on the found data, you will experiment with different adaptation approaches. Your next task is to implement several adaptation approaches and compare their results. We will start with full fine-tuning, i.e., adapt all pre-trained weights of our model on the new dataset(s). During adapatation, keep track of the language modeling performance on the other languages listed in [`task3.py`](../scripts/task3.py). What do you observe? Save your best checkpoint and re-run the hidden space exploration from `Task2`, what do you observe? 


3. Next, implement the following parameter-efficient fine-tuning approaches (you have to implement them yourself, don't use a library here!):

    - [bitfit](https://arxiv.org/abs/2106.10199)
    - [LoRA](https://arxiv.org/abs/2106.09685)
    - iA3 (see Section 3.3. of this [paper](https://arxiv.org/abs/2205.05638))

   Adapt `XGLM` on the data collected in step 1 and compare the runtime and the language modeling performance of the resutling models. Discuss your results and findings (also in comparison to the full fine-tuning approach). What are the pros and cons of each approach?


**NOTES**:

- Use [Weights & Biases (wandb)](https://wandb.ai/) to log and monitor training and evaluation metrics.
