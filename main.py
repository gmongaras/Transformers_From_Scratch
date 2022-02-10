from Transformer import transformer
import torch
from torch import nn
import re



# Add <START> and <END> stop words to the sentences
def addSTARTAndEND(inp):
    # Iterate over all sentences
    for i in range(0, len(inp)):
        inp[i] = ["<START>"] + inp[i] + ["<END>"]
    
    return inp



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





def train():
    # Hyperparameters
    inputEmbeddingSize = 64
    outputEmbeddingSize = 64
    attention_heads = 8
    keySize = 64
    querySize = keySize
    valueSize = 64
    numBlocks = 6
    alpha = 0.001
    batchSize = 10
    warmupSteps = 4000
    
    
    # Other parameters
    maxSentenceSize = 140
    
    
    
    # Other variables
    inputFileName = "data2/english_sub.txt"
    outputFileName = "data2/spanish_sub.txt"


    
    
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
    
    # Add <START> and <END> stop words to the sentence
    inputs = addSTARTAndEND(inputs)
    outputs = addSTARTAndEND(outputs)
    
    # Get vocabs for the input and output
    inputVocab = createVocab(inputs)
    outputVocab = createVocab(outputs)
    
    # Get the size of each vocab
    inputVocabSize = len(inputVocab.keys())
    outputVocabSize = len(outputVocab.keys())
    
    
    
    
    
    ### Training The Model ###
    # Create a transformer model
    model = transformer(maxSentenceSize, inputVocab, outputVocab, warmupSteps, inputEmbeddingSize, outputEmbeddingSize, attention_heads, keySize, querySize, valueSize, numBlocks, batchSize, alpha)
    
    model(inputs, outputs)


if __name__=='__main__':
    train()
