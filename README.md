# Duplicate_Question_Pairs
Check if a question pair is similar or not?

This project was made as a part of kaggle competition.

#### -- Project Timeline: [January 2022]
#### -- Project Status: [Completed]


## Problem Statement (taken from [kaggle](https://www.kaggle.com/c/quora-question-pairs/overview))
Where else but Quora can a physicist help a chef with a math problem and get cooking tips in return? Quora is a place to gain and share knowledge—about anything. It’s a platform to ask questions and connect with people who contribute unique insights and quality answers. This empowers people to learn from each other and to better understand the world.

Over 100 million people visit Quora every month, so it's no surprise that many people ask similarly worded questions. Multiple questions with the same intent can cause seekers to spend more time finding the best answer to their question, and make writers feel they need to answer multiple versions of the same question. Quora values canonical questions because they provide a better experience to active seekers and writers, and offer more value to both of these groups in the long term.

Currently, Quora uses a Random Forest model to identify duplicate questions. In this competition, Kagglers are challenged to tackle this natural language processing problem by applying advanced techniques to classify whether question pairs are duplicates or not. Doing so will make it easier to find high quality answers to questions resulting in an improved experience for Quora writers, seekers, and readers.


## Built With

- Python v3.7.10 or above
- Numpy, Pandas
- PyTorch v1.9.0 or above
- CUDA 10.2 for faster training
- Kaggle Kernels

## Tech Stack
- Glove Embeddings
- LSTM

## Dataset
The dataset can be found [here](https://www.kaggle.com/c/quora-question-pairs/data)


### Dataset Description
The train csv file contains
- id - the id of a training set question pair
- qid1, qid2 - unique ids of each question (only available in train.csv)
- question1, question2 - the full text of each question
- is_duplicate - the target variable, set to 1 if question1 and question2 have essentially the same meaning, and 0 otherwise.

The test csv file contains
- id - the id of a testing set question pair
- question1, question2 - the full text of each question


## Getting Started
Clone the repository into a local machine using

<pre>
git clone https://github.com/palash04/Duplicate_Question_Pairs.git
</pre>


## Instructions to run

In the terminal/command prompt run the following command to train the model</br>
<pre>
python3 train.py --epochs=50 --batch_size=32 --num_workers=8 --learning_rate=1e-2 --gpuidx=2 --sample_rows_per_class=100000 --train_model=1 --test_model=1
</pre>

The best model tar file `best_model.pth.tar` will be stored in the current directory.

Use the best model to run the test dataset.


## Results 
|Hyperparameters     |  Test Dataset Log Loss   | 
|---------|-----------------|
| EPOCHS=50; BATCH_SIZE=32; LEARNING_RATE=1e-2; OPTIMIZER=SGD | 0.69571 |
| EPOCHS=70; BATCH_SIZE=32; LEARNING_RATE=2e-3; OPTIMIZER=Adam | 0.53696 |

Note: Minimum log loss refers to better model.


## Authors
- Palash Mahendra Kamble - [palash04](https://github.com/palash04/)
