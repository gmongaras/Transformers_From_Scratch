from Transformer import transformer
import torch
from torch import nn



# Create a vocabulary given a 2-dimensional array of words
def createVocab(input):
    # The vocab to add items to
    vocab = {}
    
    # The current number of words in the vocab
    numWords = 0
    
    # Iterate over all words in the 2-dimensional array
    for i in input:
        for word in i:
            # If the word is not in the vocab, add it
            if word not in vocab.keys():
                vocab.update({word: numWords})
                numWords += 1
    
    # Return the vocab
    return vocab





def train():
    # Hyperparameters
    inputEmbeddingSize = 10
    outputEmbeddingSize = 10
    attention_heads = 8
    keySize = 10
    querySize = keySize
    valueSize = 10
    numBlocks = 2
    
    
    # Other parameters
    maxSentenceSize = 512
    
    
    
    # Other variables
    inputFileName = "data/english_sub.txt"
    outputFileName = "data/spanish_sub.txt"
    
    
    
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
    inputs = [i.replace("\xa0", " ").split(" ") for i in inputs]
    outputs = [i.split(" ") for i in outputs]
    
    # Get vocabs for the input and output
    inputVocab = createVocab(inputs)
    outputVocab = createVocab(outputs)
    
    # Get the size of each vocab
    inputVocabSize = len(inputVocab.keys())
    outputVocabSize = len(outputVocab.keys())
    
    
    
    
    
    ### Training The Model ###
    # Create a transformer model
    model = transformer(maxSentenceSize, inputVocab, outputVocab, inputEmbeddingSize, outputEmbeddingSize, attention_heads, keySize, querySize, valueSize, numBlocks)
    
    preds = model(inputs)
    print(preds)



train()