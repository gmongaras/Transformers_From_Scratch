import torch
from torch import nn
from torch import optim
import numpy as np




class transformer(nn.Module):
    # Inputs:
    #   maxSentenceSize - The maximum length of a sentence to accept
    #   inputVocab - Vocabulary of the input
    #   outputVocab - Vocabulary of the output
    #   inputEmbeddingSize - The size of each input embedding
    #   outputEmbeddingSize - The size of each output embedding
    #   attention_heads - The number of attention heads to use
    #                     in the multi-head attention layer
    #   keySize - The vector size of the key
    #   querySize - The vector size of the query
    #   valueSize - The vector size of the value
    def __init__(self, maxSentenceSize, inputVocab, outputVocab, inputEmbeddingSize, outputEmbeddingSize, attention_heads, keySize, querySize, valueSize):
        super(transformer, self).__init__()
        
        # Store the maximum length
        self.maxSentenceSize = maxSentenceSize
        
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
        
        # Store the number of attention heads
        self.attention_heads = attention_heads
        
        # Store the query, key, and value sizes
        self.querySize = querySize
        self.keySize = keySize
        self.valueSize = valueSize
        
        # Randomly create "attention_heads" number of key, value, 
        # and query weights. Each matrix is of shape
        # ("inputVocabSize", "querySize") or ("inputVocabSize", "valueSize")
        for i in range(0, attention_heads):
            self.keyWeights = torch.tensor(np.random.randint(0, max(self.inputVocabSize, self.outputVocabSize), size=(inputEmbeddingSize, querySize)), requires_grad=True, dtype=torch.float64)
            self.valueWeights = torch.tensor(np.random.randint(0, max(self.inputVocabSize, self.outputVocabSize), size=(inputEmbeddingSize, keySize)), requires_grad=True, dtype=torch.float64)
            self.queryWeights = torch.tensor(np.random.randint(0, max(self.inputVocabSize, self.outputVocabSize), size=(inputEmbeddingSize, valueSize)), requires_grad=True, dtype=torch.float64)
        
        # Create the weight matrix to convert the multi-head attention
        # to a single usable. The weight matrix is of the following
        # shape: (valueSize, attention_heads*maxSentenceSize)
        self.weightMatrix = torch.tensor(np.random.uniform(0, max(self.inputEmbeddingSize, self.outputVocabSize), size=(maxSentenceSize, attention_heads*maxSentenceSize)), requires_grad=True, dtype=torch.float64)
    
    
    
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
            
            # pad the array with "PAD" values.
            embedding = torch.tensor(np.pad(embedding.numpy(), ((0, self.maxSentenceSize - embedding.shape[0]), (0, 0)), mode='constant', constant_values=max(self.inputVocabSize, self.outputVocabSize)+1), requires_grad=False)
            
            # Get a poitional encoding vector which is the same
            # size as the sentence embedding
            posEnc = np.array([[np.sin(i/np.power(10000, (2*embedding[i].shape[0])/self.inputEmbeddingSize))] if (i%2 == 0) else [np.cos(i/np.power(10000, (2*embedding[i].shape[0])/self.inputEmbeddingSize))] for i in range(0, embedding.shape[0])])
            
            # Apply positional encodings to the embedding
            embedding += posEnc
            
            # Add the embedding to the embeddings array
            embeddings.append(embedding)
        
        # Return the embeddings
        return torch.stack(embeddings)
    
    
    
    # Given some embeddings, the self-attention layer
    # computes the self-attention for the embeddings
    # Inputs:
    #   embeddings - The embeddings to compute the self-attention for
    def selfAttention(self, embeddings):
        try:
            keys = torch.matmul(embeddings, self.keyWeights)
            values = torch.matmul(embeddings, self.valueWeights)
            queries = torch.matmul(embeddings, self.queryWeights)
        except:
                keys = torch.matmul(embeddings, self.keyWeights.T)
                values = torch.matmul(embeddings, self.valueWeights.T)
                queries = torch.matmul(embeddings, self.queryWeights.T)
        return torch.matmul(nn.functional.softmax(torch.matmul(queries, keys.reshape(keys.shape[0], keys.shape[2], keys.shape[1]))/int(np.sqrt(self.keySize)), dim=-1), values)
    

    
    # Given some embeddings, the multi-head attention layer
    # computes the multihead attention for the embeddings
    # Inputs:
    #   embeddings - The embeddings to compute multi-head attention for
    def multiHeadAttention(self, embeddings):
        # Holds "attention_heads" number of self-attention
        attentionValues = []
        
        # Collect the self-attention
        for i in range(0, self.attention_heads):
            # Calculate the self-attention for all
            # given embeddings
            attentionVals = self.selfAttention(embeddings)
            # for embedding in embeddings:
            #     attentionVals.append(self.selfAttention(embedding))
            attentionValues.append(attentionVals)
            
        # Convert the list of attention to a tensor
        attentionValues = torch.stack(attentionValues)
        
        # Reshape the tensor to a workable shape
        attentionValues = attentionValues.reshape((attentionValues.shape[1], attentionValues.shape[0], attentionValues.shape[2], attentionValues.shape[3]))
        attentionValues = attentionValues.reshape((attentionValues.shape[0], attentionValues.shape[1]*attentionValues.shape[2], attentionValues.shape[3]))
        #attentionValues = attentionValues.reshape((int(attentionValues.shape[0]/self.attention_heads), int(attentionValues.shape[1]*self.attention_heads), attentionValues.shape[2]))
        
        # Multiply the attention values by the weight matrix
        finalAttention = torch.matmul(self.weightMatrix, attentionValues)
        
        # Return the final attention values
        return finalAttention
    
    
    # Translate the given sentences
    # Input:
    #   x - The batch of sentences to translate
    def forward(self, x):
        # Send the words through the word embedding layer
        embeddings = self.embedBatch(x)
        
        # Compute the multi-head attention for the embeddings
        att = self.multiHeadAttention(embeddings)
        att = self.multiHeadAttention(att)
        att = self.multiHeadAttention(att)
        att = self.multiHeadAttention(att)
        print()