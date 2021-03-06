from Transformer import transformer
from Transformer import addSTARTAndEND
import re
import matplotlib.pyplot as plt
import numpy as np
import os
#torch.autograd.set_detect_anomaly(True)



# Create a vocabulary given a 2-dimensional array of words
def createVocab(inp):
    # The vocab to add items to
    vocab = {}
    
    # The current number of words in the vocab
    numWords = 0
    
    # Iterate over all words in the 2-dimensional array
    for i in inp:
        for word in i:
            # If the word is not in the vocab, add it
            if word not in vocab.keys():
                vocab.update({word: numWords})
                numWords += 1
    
    # Add a the special <PAD> character to the vocab
    vocab.update({"<PAD>": numWords})
    
    # Return the vocab
    return vocab





def main():
    # Hyperparameters
    inputEmbeddingSize = 20         # Embedding size of the inputs
    outputEmbeddingSize = 20        # Embedding size of the outputs
    attention_heads = 4             # Number of attention heads for multi-head attention
    keySize = 16                    # Size of the key for multi-head attention
    querySize = keySize             # Size of the query for multi-head attention
    valueSize = 16                  # Size of the value for multi-head attention
    numBlocks = 4                   # Number of transformer blocks to use
    batchSize = 20                  # Size of each minibatch
    warmupSteps = 600               # Number of warmup steps to train the model
    numSteps = 1000                 # Total number of steps to train the model
    maxSentenceSize = 140           # The max size of each sentence
    clipVal = 100000                # The bound used to clip the gradients
    
    
    
    # File variables
    inputFileName = "data2/english_sub.txt"         # Name of the input data file
    outputFileName = "data2/spanish_sub.txt"        # Name of the output data file
    lossPlotFileName = "visualizations/plot.png"    # Name of the file to save the loss plot to
    modelSaveName = "models/modelCkPt.pt"           # Name of the file to save the model to
    modelLoadName = "models/modelCkPt_100E_6L.pt"   # Name of the file to load the model from



    # Other parameters
    loadModel = False        # True to load the model from "modelLoadName", False otherwise
    trainModel = True        # True to train the model, False otherwise
    stepsToSave = 5          # Number of steps till model is saved
    startStep = 1            # Step to start on when training (use 1 if not using pretrained model)
    
    
    # Testing the model
    testData = np.array(["Implementation of the President to enhance the Russian Federation.",
                         "This is another test sentence.",
                         "The high debt burden has weakened the capacity of many Governments to service their increasing external debt and has eroded resources available for social development."])
        
    
    
    ### Reading The Data ###
    # Open the input and output file for reading
    inputFile = open(inputFileName, "r", encoding='utf8')
    outputFile = open(outputFileName, "r", encoding='utf8')
    
    # Read the input and output file into memory
    inputs = inputFile.read().split("\n")
    outputs = outputFile.read().split("\n")
    
    # Close the files
    inputFile.close()
    outputFile.close()  
    
    
    
    
    ### Clean the inputs and outputs ###
    # Split all the sentences into word tensors and clean them
    inputs = [re.sub(r'[^\w\s]', '', i.replace("\xa0", " ")).lower().split(" ") for i in inputs]
    outputs = [re.sub(r'[^\w\s]', '', i.replace("\xa0", " ")).lower().split(" ") for i in outputs]
    
    # Add <START> and <END> stop words to the data
    inputs = addSTARTAndEND(inputs)
    outputs = addSTARTAndEND(outputs)
    
    # Get vocabs for the input and output
    inputVocab = createVocab(inputs)
    outputVocab = createVocab(outputs)
    
    
    
    
    
    ### Training The Model ###
    # Create a transformer model
    model = transformer(maxSentenceSize, inputVocab, outputVocab, warmupSteps, inputEmbeddingSize, outputEmbeddingSize, attention_heads, keySize, querySize, valueSize, numBlocks, batchSize, stepsToSave, startStep)
    
    # Load the model if specified
    if loadModel:
        model.loadModel(modelLoadName)
    
    # Train the model if specified
    if trainModel:
        losses, stepCounts = model.trainModel(inputs, outputs, numSteps, modelSaveName, clipVal)
    
        # Create a graph of the model training process and save it
        plt.plot(stepCounts, losses)
        plt.xlabel("Number of Model Updates")
        plt.ylabel("Loss at step")
        plt.title("Number of Model Updates vs. Model Loss")
        graphDir = "/".join(lossPlotFileName.split("/")[0:-1])
        if not os.path.isdir(graphDir):
            os.mkdir(graphDir)
        plt.savefig(lossPlotFileName)
    
    ### Testing The Model ###
    output = model(testData)
    
    print("---------------------------------------------------------")
    print("Translations:")
    for i in range(0, len(testData)):
        v = output[i] + ["<END>"]
        print(f"English: {testData[i]}")
        print(f"Spanish: {' '.join(v[v.index('<START>')+2:v.index('<END>')])}")
        print()


if __name__=='__main__':
    main()
