# Venue Accessibility with Google Reviews 

This project aims to highlight the perspective of people with Reduced Mobility (RM) living in Amsterdam via analysis of public venue reviews. This is done using Natural Language Processing (NLP) techniques such as Aspect Based Sentiment Analysis and Opinion Summarisation. 

It is an extension of the work carried out by [L. Da Rocha Bazilio](https://github.com/Amsterdam-Internships/Venue-Accessibility-NLP), to understand how different models are able to extract aspects in reference to accessibility with data that is noisy e.g with explicit aspects to classify. 

In addition, understanding the impact of Opinion Summarisation on the reviews about accessibility. Also, how this can make activity and journey planning for those with RM easier. 


This is an example of the UI of the application that pipeline will be connected to:
![](media/examples/venue-accessibility-example-profile-cropped.png)


---


## Project Folder Structure

There are the following folders in the structure:

1) [`datasets`](./datasets): This is where you should place your data for training and testing.
1) [`media`](./media): This is where results of each step of the pipeline are stored as images.
1) [`notebooks`](./notebooks): This contains the notebooks of th pipeline.
1) [`results`](./results): Here you will find the txt form of the results.
1) [`src`](./src): Folder for all source files specific to this project
1) [`scripts`](./scripts): Folder with example scripts for performing different tasks (could serve as usage documentation)
1) [`tests`](./tests) Here I store all of the tests for project
---


## Installation


1) Clone this repository:
    ```bash
    git clone git@github.com:Amsterdam-Internships/Venue-Accessibility-Google-Reviews.git
    ```

2) Install all dependencies:
    ```bash
    conda install environment.yaml
    ```
---


## Usage

Explain example usage, possible arguments, etc. E.g.:

To train... 


```
$ python train.py --some-importang-argument
```

If there are too many command line arguments, you can add a nice table with explanation (thanks, [Diana Epureano](https://www.linkedin.com/in/diana-epureanu-235104153/)!)

|Argument | Type or Action | Description | Default |
|---|:---:|:---:|:---:|
|`--batch_size`| int| `Batch size.`|  32|
|`--device`| str| `Training device, cpu or cuda:0.`| `cpu`|
|`--early-stopping`|  `store_true`| `Early stopping for training of sparse transformer.`| True|
|`--epochs`| int| `Number of epochs.`| 21|
|`--input_size`|  int| `Input size for model, i.e. the concatenation length of te, se and target.`| 99|
|`--loss`|  str|  `Type of loss to be used during training. Options: RMSE, MAE.`|`RMSE`|
|`--lr`|  float| `Learning rate.`| 1e-3|
|`--train_ratio`|  float| `Percentage of the training set.`| 0.7|
|...|...|...|...|


Alternatively, as a way of documenting the intended usage, you could add a `scripts` folder with a number of scripts for setting up the environment, performing training in different modes or different tasks, evaluation, etc (thanks, [Tom Lotze](https://www.linkedin.com/in/tom-lotze/)!)

---


## How it works

You can explain roughly how the code works, what the main components are, how certain crucial steps are performed...

---
## Acknowledgements


This work and code is based off of the priort work of @Lizzydrb .
