import torch
from torch import nn
from torch import optim
import numpy as np




class transformer(nn.Module):
    # Inputs:
    #   inputVocab - Vocabulary of the input
    #   outputVocab - Vocabulary of the output
    #   inputEmbeddingSize - The size of each input embedding
    #   outputEmbeddingSize - The size of each output embedding
    def __init__(self, inputVocab, outputVocab, inputEmbeddingSize, outputEmbeddingSize):
        super(transformer, self).__init__()
        
        # Store the input and output vocabulary
        self.inputVocab = inputVocab
        self.outputVocab = outputVocab
        
        # Store the input and output vocabulary sizes
        self.inputVocabSize = len(inputVocab.keys())
        self.outputVocabSize = len(outputVocab.keys())
        
        # Store the input and output embedding sizes
        self.inputEmbeddingSize = inputEmbeddingSize
        self.outputEmbeddingSize = outputEmbeddingSize
        
        # The word embedding layer for the inputs
        self.input_embedding_layer = nn.Embedding(self.inputVocabSize, inputEmbeddingSize)
        
        # The word embedding layer for the output
        self.output_embedding_layer = nn.Embedding(self.outputVocabSize, outputEmbeddingSize)
    
    
    
    # Translate a list of words into word embeddings
    # Inputs:
    #   words - A 1-dimensional array of words to embed
    #   vocabToUse - "Input" to use the input vocabulary. Another 
    #                other string to use the output vocabulary
    def embedWords(self, words, vocabToUse):
        if vocabToUse == "Input":
            # Iterate over every word and translate it to a numerical value
            words_num = []
            for word in words:
                try:
                    words_num.append(self.inputVocab[word])
                except:
                    words_num.append(self.inputVocabSize)
            words_num = torch.tensor(words_num, requires_grad=False)
            
            # Embed the words
            embeddings = self.input_embedding_layer(words_num).detach()
            
        else:
            # Iterate over every word and translate it to a numerical value
            words_num = []
            for word in words:
                try:
                    words_num.append(self.outputVocab[word])
                except:
                    words_num.append(self.outputVocabSize)
            words_num = torch.tensor(words_num, requires_grad=False)
            
            # Embed the words
            embeddings = self.input_embedding_layer(words_num).detach()
        
        # Return the embeddings
        return embeddings

    
    # Embed a batch of sentences to a batch of word embeddings
    # Inputs:
    #   batch - The batch of sentences to embed
    def embedBatch(self, batch):
        # Send the words through the word embedding layer
        embeddings = []
        for sentence in batch:
            # Get the embedding
            embedding = self.embedWords(sentence, "Input")
            
            # Get a poitional encoding vector which is the same
            # size as the sentence embedding
            posEnc = np.array([[np.sin(i/np.power(10000, (2*embedding[i].shape[0])/self.inputEmbeddingSize))] if (i%2 == 0) else [np.cos(i/np.power(10000, (2*embedding[i].shape[0])/self.inputEmbeddingSize))] for i in range(0, embedding.shape[0])])
            
            # Apply positional encodings to the embedding
            embedding += posEnc
            
            # Add the embedding to the embeddings array
            embeddings.append(embedding)
        
        # Return the embeddings
        return embeddings
    
    
    # Translate the given sentences
    # Input:
    #   x - The batch of sentences to translate
    def forward(self, x):
        # Send the words through the word embedding layer
        embeddings = self.embedBatch(x)
        print()