# Exploring multi-lingual representation spaces

1. Now that you are familiar with how to load a dataset and run inference using a pre-trained model, the goal of this exercise is to obtain representations from a model and visualize them. Using the same model and data as in `Task1`, your task is to obtain the hidden representations from every layer of the model for each input token of every sequence (except the padding tokens). You will have to save these representations on disk using [HDF5](https://docs.h5py.org/en/stable/index.html). Use the [`task2.py`](../scripts/task2.py) file for your implementation. In addition to the token representations, you will also have to implement the option to save a sentence representation by mean-pooling over the individual token representations. Again, make sure to ingore the padding tokens. When saving representations via hdf5, make sure to also store the corresponding tokens and sequences.

    **NOTES**: 

    - Given the large size of the hidden repesentation vectors, we will limit ourselves to a random subset of 200 sentences for each language for this task. 

    - Make use of the CS cluster for this task. Keep an eye on the size of the files you are saving as well as the number of jobs you are submitting. Make sure to delete runs that are on hold.


2. Once you have saved the layer-wise hidden representations per language, your next task is to visualize the representation space of our model (for both the token and sentence representations). To do that, you need to project the high dimensional hidden representations to a lower dimenstional space. PCA and tSNE are suitable approaches to do that. To get started, apply PCA to the hidden representations of each language from the first layer of the model and visualize the result in 2D. Repeat the same with tSNE and compare to the results you got with PCA. Discuss what you observe. Next, repeat the visualization (PCA and tSNE) for each layer of the model (including the embedding layer). Discuss what you observe.

    **NOTES**: 

    - Annotate the projected token or sentence representations with their correpoding string representation (the actual token or sequence). [See here for a nice example](https://altair-viz.github.io/gallery/scatter_tooltips.html). 

    - [This paper](https://aclanthology.org/2020.vardial-1.12/) is a good starting point to read about the analysis of multi-lingual representation spaces.
