# General Project Information
<strong>Title</strong>: Transformers From Scratch
<strong>Data Created</strong>: January 25, 2022

<strong>Data From</strong>: https://www.statmt.org/wmt12/translation-task.html

# Project Description
In this project, I aim to build a transformer model from scratch based on the "Attention Is All You Need" paper:

https://arxiv.org/pdf/1706.03762.pdf

Using PyTorch, I am going to construct a model that will be able to translate an English sentence to a Spanish sentence.

# Project Requirements
Below are the following python library requirements to be able to run the project
- PyTorch - 1.8.1
- NumPy - 1.19.5
- Matplotlib - 3.4.2

Note: The library version I used are listed as well to ensure the model can be run successfully on other computers.

# Project Execution
To execute this project, download the repository and use the following command to run the project:

`python3 main.py`

## Hyperparameters
There are many hyperparameters that can be tuned in the model, the hyperparameters can be found at the beginning of the main function in main.py. Please note that the larger the keys, values, embeddings, etc., the more memory the model will require to train.

## Training The Model
To train the model, set the flag on line 65 named `trainModel` to true. 

The number of steps can be specified through the `numSteps` variable in the hyperparameter section of the main function and specifies the number of times the model should be updated.

As the model is being trained, the model will be saved after every 5 steps by default. The number of steps after the model is saved can be specified with the `stepsToSave` in the <strong>Other parameters</strong> section of the main function.

Additionally, a graph of the model will be saved at the end of training. The graph shows the relationship between the number of steps and the loss at that current step. By default, the graph will be saved to the file named specified by the `lossPlotFileName` variable in the <strong>File variables</strong> section of the main function.

After each step, the model will output some text to the console showing its progress.
- The first line specifies the step number
- The second line is the first 10 words in an English training sentence
- The third line is the first 10 words in a Spanish training sentence (what we want the model to predict)
- The fourth line is the first 10 words the model predict on the English training sentence

The output looks as follows:

<img width="392" alt="image" src="https://user-images.githubusercontent.com/43501738/155207336-3af01b76-9cc0-43de-8c09-e61a15ea2935.png">

## Testing The Model
When the model is finished training, it will be tested on a couple of sentences. By default, the test sentences are:
- Implementation of the President to enhance the Russian Federation
- This is a test sentence

To use different test sentences, change the array named `testData` in the <bold>Testing the model</bold> section of the main function. Each line in the array should be a new test sentence.

To skip training and go straight to testing, the `trainModel` flag can be changed to False in the <strong>Other parameters</strong> section of the main function.

## Saving And Loading A Model
By default, the model will be saved to the `models/` directory and will be saved to a file named `modelCkPt.pt`. This path can be changed using the `modelSaveName` variable in the <strong>File variables</strong> section of the main function.

As stated in the <strong>Training The Model</strong> section, the model will be saved every 5 steps by default. This parameter can be changed by specifying the number of steps until the model is saved using the `stepsToSave` variable in the <strong>Other parameters</strong> section of the main function.

To load a model, change the `loadModel` flag in the <strong>Other parameters</strong> to true and change the `modelLoadName` variable in the <strong>File variables</strong> section to the path of the file you want to load in.

<strong>NOTE: The hyperparameters at the top of the main function must be the same as the hyperparameters of the model being loaded in</strong>

# Results
Unfortunately, I do not have the hardware required to train the model on a large dataset, so I trained it on a small sample of the data, which can be found in the `data2` directory of this repo. After about 1000 steps, the model pretty much mastered the small dataset, as shown below.


### Beginning of Training
Here's the progress the model made during the beginning of it's training
![Beginning of Training](https://github.com/gmongaras/Transformers_From_Scratch/blob/main/Progress%20Images/Beginning_Of_Training.png)

### Middle of Training
Here's the progress the same model made during the middle of it's training
![Middle of Training](https://github.com/gmongaras/Transformers_From_Scratch/blob/main/Progress%20Images/Middle_Of_Training.png)

### End of Training
Here's the progress the model made at the end of it's training on the training dataset
![End of Training](https://github.com/gmongaras/Transformers_From_Scratch/blob/main/Progress%20Images/End_Of_Training.png)

And here's the translations on a few test sentences
![Test_Results](https://github.com/gmongaras/Transformers_From_Scratch/blob/main/Progress%20Images/Test_Results.png)

As you can see, the model did really good on the training examples and practically perfectly translated all training sentences. As for the test dataset, the model clearly didn't get the predictions correct, but this problem is due to the small training smaple I used. If it used all the data, the model will likely be able to effectively translate these sentences along with new ones it sees.
