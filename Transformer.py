import torch
from torch import nn
from torch import optim
import numpy as np



class multiHeadAttention(nn.Module):
    # Inputs:
    #   maxSentenceSize - The maximum length of a sentence to accept
    #   inputVocabSize - Vocabulary size of the input
    #   outputVocabSize - Vocabulary size of the output
    #   inputEmbeddingSize - The size of each input embedding
    #   outputEmbeddingSize - The size of each output embedding
    #   attention_heads - The number of attention heads to use
    #                     in the multi-head attention layer
    #   keySize - The vector size of the key
    #   querySize - The vector size of the query
    #   valueSize - The vector size of the value
    #   alpha - The learning rate of the model
    def __init__(self, maxSentenceSize, inputVocabSize, outputVocabSize, inputEmbeddingSize, outputEmbeddingSize, attention_heads, keySize, querySize, valueSize, alpha=0.001):
        super(multiHeadAttention, self).__init__()
        
        # Store the number of attention heads
        self.attention_heads = attention_heads
        
        # Store the query, key, and value sizes
        self.querySize = querySize
        self.keySize = keySize
        self.valueSize = valueSize
        
        # Randomly create "attention_heads" number of key, value, 
        # and query weights. Each matrix is of shape
        for i in range(0, attention_heads):
            self.keyWeights = torch.tensor(np.random.randint(0, max(inputVocabSize, outputVocabSize), size=(inputEmbeddingSize, querySize)), requires_grad=True, dtype=torch.float64)
            self.valueWeights = torch.tensor(np.random.randint(0, max(inputVocabSize, outputVocabSize), size=(inputEmbeddingSize, keySize)), requires_grad=True, dtype=torch.float64)
            self.queryWeights = torch.tensor(np.random.randint(0, max(inputVocabSize, outputVocabSize), size=(inputEmbeddingSize, valueSize)), requires_grad=True, dtype=torch.float64)
        
        # Create the weight matrix to convert the multi-head attention
        # to a single usable. The weight matrix is of the following
        # shape: (maxSentenceSize, attention_heads*maxSentenceSize)
        self.weightMatrix = torch.tensor(np.random.uniform(0, max(inputVocabSize, outputVocabSize), size=(attention_heads*valueSize, inputEmbeddingSize)), requires_grad=True, dtype=torch.float64)
        
        # The optimizer for the model
        self.optimizer = optim.Adam([self.keyWeights, self.valueWeights, self.queryWeights, self.weightMatrix], lr=alpha)
    
    
    # Given some embeddings, the self-attention layer
    # computes the self-attention for the embeddings
    # Inputs:
    #   embeddings - The embeddings to compute the self-attention for
    #   embeddings2 - An optional second set of embeddings used to
    #                 compute part of the KVQ values
    def selfAttention(self, embeddings, embeddings2 = None):
        embeddings = embeddings.double()
        try:
            keys = torch.matmul(embeddings, self.keyWeights)
            values = torch.matmul(embeddings, self.valueWeights)
            if embeddings2 != None:
                embeddings2 = embeddings2.double()
                queries = torch.matmul(embeddings2, self.queryWeights)
            else:
                queries = torch.matmul(embeddings, self.queryWeights)
        except:
            keys = torch.matmul(embeddings.double(), self.keyWeights.T)
            values = torch.matmul(embeddings, self.valueWeights.T)
            if embeddings2 != None:
                embeddings2 = embeddings2.double()
                queries = torch.matmul(embeddings2, self.queryWeights.T)
            else:
                queries = torch.matmul(embeddings, self.queryWeights.T)
        return torch.matmul(nn.functional.softmax(torch.matmul(queries, keys.reshape(keys.shape[0], keys.shape[2], keys.shape[1]))/int(np.sqrt(self.keySize)), dim=-1), values)
    
    
    
    # Given some embeddings, the multi-head attention layer
    # computes the multihead attention for the embeddings
    # Inputs:
    #   embeddings - The embeddings to compute multi-head attention for
    #   embeddings2 - An optional second set of embeddings used to
    #                 compute part of the KVQ values
    def forward(self, embeddings, embeddings2 = None):
        # Holds "attention_heads" number of self-attention
        attentionValues = []
        
        # Collect the self-attention
        for i in range(0, self.attention_heads):
            # Calculate the self-attention for all
            # given embeddings
            attentionVals = self.selfAttention(embeddings, embeddings2)
            # for embedding in embeddings:
            #     attentionVals.append(self.selfAttention(embedding))
            attentionValues.append(attentionVals)
            
        # Convert the list of attention to a tensor
        attentionValues = torch.stack(attentionValues)
        
        # Reshape the tensor to a workable shape
        #attentionValues = attentionValues.reshape((attentionValues.shape[1], attentionValues.shape[0], attentionValues.shape[2], attentionValues.shape[3]))
        attentionValues = attentionValues.reshape((attentionValues.shape[1], attentionValues.shape[2], attentionValues.shape[3]*attentionValues.shape[0]))
        #attentionValues = attentionValues.reshape((int(attentionValues.shape[0]/self.attention_heads), int(attentionValues.shape[1]*self.attention_heads), attentionValues.shape[2]))
        
        # Multiply the attention values by the weight matrix
        finalAttention = torch.matmul(attentionValues, self.weightMatrix)
        
        # Return the final attention values
        return finalAttention




class FeedForward(nn.Module):
    # Feed forward network with the following layers
    # - Fully Connected Layer
    # - ReLU Layer
    # - Fully Connected Layer
    # - ReLU Layer
    # Inputs:
    #   InputDim - Dimensions of the input
    #   inderDim - Dimension of the inner shape
    #   outputDim - Dimensions of the output
    #   alpha - The learning rate of the model
    def __init__(self, inputDim=512, innerDim=2048, outputDim=512, alpha=0.001):
        super(FeedForward, self).__init__()
        
        # The fully connected model
        self.model = nn.Sequential(
            nn.Linear(inputDim, innerDim),
            nn.ReLU(),
            nn.Linear(innerDim, outputDim),
            nn.ReLU(),
        )
        
        # The optimizer for the model
        self.optimizer = optim.Adam(self.model.parameters())
    
    
    
    # Inputs:
    #   input - The value to put through the network
    def forward(self, input):
        return self.model(input)




class inputTransformerBlock(nn.Module):
    # Inputs:
    #   maxSentenceSize - The maximum length of a sentence to accept
    #   inputVocabSize - Size of input vocabulary
    #   outputVocabSize - Size of output vocabulary
    #   inputEmbeddingSize - The size of each input embedding
    #   outputEmbeddingSize - The size of each output embedding
    #   attention_heads - The number of attention heads to use
    #                     in the multi-head attention layer
    #   keySize - The vector size of the key
    #   querySize - The vector size of the query
    #   valueSize - The vector size of the value
    def __init__(self, maxSentenceSize, inputVocabSize, outputVocabSize, inputEmbeddingSize, outputEmbeddingSize, attention_heads, keySize, querySize, valueSize):
        super(inputTransformerBlock, self).__init__()
        
        # Store some required parameters
        self.maxSentenceSize = maxSentenceSize
        self.valueSize = valueSize
        
        # Create the multi-head attention layer
        self.multiHead_Attention = multiHeadAttention(maxSentenceSize, inputVocabSize, outputVocabSize, inputEmbeddingSize, outputEmbeddingSize, attention_heads, keySize, querySize, valueSize)
        
        # Create a fully connected layer
        self.FullyConnected = FeedForward()
    
    
    # Get the transformer encodings for the inputs
    # Inputs:
    #   inputEncoding - The encoded input to send through a
    #                   transformer block
    # Outputs:
    #   FC_norm - The results of sending the inputs through the
    #             input transformer block
    def forward(self, inputEncoding):
        # Compute the multi-head attention for the inputs
        att = self.multiHead_Attention(inputEncoding)
        
        # Send the attiontion through an add and norm layer
        att_add = att + inputEncoding
        att_norm = nn.LayerNorm((self.maxSentenceSize, self.valueSize))(att_add.float())
        
        # Compute the forward value
        norm_res = att_norm.reshape(att_norm.shape[0], att_norm.shape[2], att_norm.shape[1])
        FC = self.FullyConnected(norm_res)
        FC = FC.reshape(FC.shape[0], FC.shape[2], FC.shape[1])
        
        # Send the fully connected output through an add and norm layer
        FC_add = FC + att_norm
        FC_norm = nn.LayerNorm((self.maxSentenceSize, self.valueSize))(FC_add)
        
        # Return the values
        return FC_norm





class outputTransformerBlock(nn.Module):
    # Inputs:
    #   maxSentenceSize - The maximum length of a sentence to accept
    #   inputVocabSize - Size of input vocabulary
    #   outputVocabSize - Size of output vocabulary
    #   inputEmbeddingSize - The size of each input embedding
    #   outputEmbeddingSize - The size of each output embedding
    #   attention_heads - The number of attention heads to use
    #                     in the multi-head attention layer
    #   keySize - The vector size of the key
    #   querySize - The vector size of the query
    #   valueSize - The vector size of the value
    def __init__(self, maxSentenceSize, inputVocabSize, outputVocabSize, inputEmbeddingSize, outputEmbeddingSize, attention_heads, keySize, querySize, valueSize):
        super(outputTransformerBlock, self).__init__()
        
        # Store some required parameters
        self.maxSentenceSize = maxSentenceSize
        self.valueSize = valueSize
        
        # Create the multi-head attention layer
        self.multiHead_Attention = multiHeadAttention(maxSentenceSize, inputVocabSize, outputVocabSize, inputEmbeddingSize, outputEmbeddingSize, attention_heads, keySize, querySize, valueSize)
        
        # Create a fully connected layer
        self.FullyConnected = FeedForward()
    
    
    # Get the transformer encodings for the output
    # Inputs:
    #   inputRes - The results of sending the inputs through the
    #              input transformer block
    #   outputEncoding - The encoded output to send through a
    #                    transformer block
    # Outputs:
    #   FC_norm - The results of sending the outputs through the
    #             output transformer block
    def forward(self, inputRes, outputEncoding):
        # Compute the multi-head attention for the outputs
        att = self.multiHead_Attention(outputEncoding)
        
        # Send the attiontion through an add and norm layer
        att_add = att + outputEncoding
        att_norm = nn.LayerNorm((self.maxSentenceSize, self.valueSize))(att_add.float())
        
        # Compute the attiontion for the input results and output results
        att2 = self.multiHead_Attention(inputRes, att_norm)
        
        # Send the new attiontion through an add and norm layer
        att2_add = att2 + outputEncoding
        att2_norm = nn.LayerNorm((self.maxSentenceSize, self.valueSize))(att2_add.float())
        
        # Compute the forward value
        norm_res = att2_norm.reshape(att2_norm.shape[0], att2_norm.shape[2], att2_norm.shape[1])
        FC = self.FullyConnected(norm_res)
        FC = FC.reshape(FC.shape[0], FC.shape[2], FC.shape[1])
        
        # Send the fully connected output through an add and norm layer
        FC_add = FC + att2_norm
        FC_norm = nn.LayerNorm((self.maxSentenceSize, self.valueSize))(FC_add)
        
        # Return the values
        return FC_norm
        





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
    #   numBlocks - The number of transformer blocks
    def __init__(self, maxSentenceSize, inputVocab, outputVocab, inputEmbeddingSize, outputEmbeddingSize, attention_heads, keySize, querySize, valueSize, numBlocks):
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
        
        # Save the value size
        self.valueSize = valueSize
        
        # The word embedding layer for the inputs
        self.input_embedding_layer = nn.Embedding(self.inputVocabSize, inputEmbeddingSize)
        
        # The word embedding layer for the output
        self.output_embedding_layer = nn.Embedding(self.outputVocabSize, outputEmbeddingSize)
        
        # Store the number of blocks
        self.numBlocks = numBlocks
        
        # Create the input blocks
        self.inputBlocks = []
        for i in range(0, int(numBlocks/2)):
            self.inputBlocks.append(inputTransformerBlock(maxSentenceSize, self.inputVocabSize, self.outputVocabSize, inputEmbeddingSize, outputEmbeddingSize, attention_heads, keySize, querySize, valueSize))
            
        # Create the output blocks
        self.outputBlocks = []
        for i in range(0, int(numBlocks/2)):
            self.outputBlocks.append(outputTransformerBlock(maxSentenceSize, self.inputVocabSize, self.outputVocabSize, inputEmbeddingSize, outputEmbeddingSize, attention_heads, keySize, querySize, valueSize))
    
    
    
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
    
    
    
    
    # Translate the given sentences
    # Input:
    #   x - The batch of sentences to translate
    def forward(self, x):
        # Send the words through the word embedding layer
        embeddings = self.embedBatch(x)
        
        # Test data
        inputRes = embeddings
        outputRes = embeddings
        
        # Send the embeddings through each layer
        for i in range(0, int(self.numBlocks/2)):
            # Send the inputs through the input block
            inputRes = self.inputBlocks[i](inputRes)
            
            # Send the output through the output block
            outputRes = self.outputBlocks[i](inputRes, outputRes)
        
        
        # Send the 