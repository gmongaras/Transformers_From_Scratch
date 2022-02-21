import re
import torch
from torch import nn
from torch import optim
import numpy as np
import os


device = torch.device('cpu')
#if torch.cuda.is_available():
#    device = torch.device('cuda')
    
    
# Add <START> and <END> stop words to the sentences
def addSTARTAndEND(inp):
    # Iterate over all sentences
    for i in range(0, len(inp)):
        inp[i] = ["<START>"] + inp[i] + ["<END>"]
    
    return inp


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
    #   mask - True if masking should be used, False otherwise
    def __init__(self, maxSentenceSize, inputVocabSize, outputVocabSize, inputEmbeddingSize, outputEmbeddingSize, attention_heads, keySize, querySize, valueSize, mask = False):
        super(multiHeadAttention, self).__init__()
        
        # Store the number of attention heads
        self.attention_heads = attention_heads
        
        # Store the query, key, and value sizes
        self.querySize = querySize
        self.keySize = keySize
        self.valueSize = valueSize
        
        # Randomly create "attention_heads" number of key, value, 
        # and query weights. Each matrix is of shape
        self.keyWeights = []
        self.valueWeights = []
        self.queryWeights = []
        for i in range(0, attention_heads):
            self.keyWeights.append(nn.Parameter(torch.tensor(np.random.uniform(0, 1, size=(inputEmbeddingSize, keySize)), requires_grad=True, dtype=torch.float64, device=device), requires_grad=True))
            self.valueWeights.append(nn.Parameter(torch.tensor(np.random.uniform(0, 1, size=(inputEmbeddingSize, keySize)), requires_grad=True, dtype=torch.float64, device=device), requires_grad=True))
            self.queryWeights.append(nn.Parameter(torch.tensor(np.random.uniform(0, 1, size=(inputEmbeddingSize, querySize)), requires_grad=True, dtype=torch.float64, device=device), requires_grad=True))
        
        # Convert the arrays to parameter lists so it can be updated
        self.keyWeights = nn.ParameterList(self.keyWeights)
        self.valueWeights = nn.ParameterList(self.valueWeights)
        self.queryWeights = nn.ParameterList(self.queryWeights)
        
        # Create the weight matrix to convert the multi-head attention
        # to a single usable. The weight matrix is of the following
        # shape: (maxSentenceSize, attention_heads*maxSentenceSize)
        self.weightMatrix = nn.Parameter(torch.tensor(np.random.uniform(0, 1, size=(attention_heads*valueSize, inputEmbeddingSize)), requires_grad=True, dtype=torch.float64, device=device), requires_grad=True)
        
        # If masking is used, create a mask which is of
        # shape (maxSentenceSize, maxSentenceSize) which
        # will force the model to not look ahead
        self.useMask = mask
        if mask:
            self.mask = mask = torch.tensor([[0 for j in range(0, i+1)] + [-np.inf for j in range(i+1, maxSentenceSize)] for i in range(0, maxSentenceSize)], requires_grad=True, device=device)
        
    
    # Given some embeddings, the self-attention layer
    # computes the self-attention for the embeddings
    # Inputs:
    #   attIndex - The attention index used to know what K, Q, and V to use
    #   embeddings - The embeddings to compute the self-attention for
    #   embeddings2 - An optional second set of embeddings used to
    #                 compute part of the KVQ values
    def selfAttention(self, attIndex, embeddings, embeddings2 = None):
        embeddings = embeddings.double()
        try:
            keys = torch.matmul(embeddings, self.keyWeights[attIndex])
            values = torch.matmul(embeddings, self.valueWeights[attIndex])
            if embeddings2 != None:
                embeddings2 = embeddings2.double()
                queries = torch.matmul(embeddings2, self.queryWeights[attIndex])
            else:
                queries = torch.matmul(embeddings, self.queryWeights[attIndex])
        except:
            keys = torch.matmul(embeddings.double(), self.keyWeights[attIndex].T)
            values = torch.matmul(embeddings, self.valueWeights[attIndex].T)
            if embeddings2 != None:
                embeddings2 = embeddings2.double()
                queries = torch.matmul(embeddings2, self.queryWeights[attIndex].T)
            else:
                queries = torch.matmul(embeddings, self.queryWeights[attIndex].T)
        
        # Calculate the attention. Use a mask if specified
        if self.useMask:
            return torch.matmul(nn.functional.softmax((torch.matmul(queries, keys.reshape(keys.shape[0], keys.shape[2], keys.shape[1])) + self.mask)/int(np.sqrt(self.keySize)), dim=-1), values)
        else:
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
            attentionVals = self.selfAttention(i, embeddings, embeddings2)
            attentionValues.append(attentionVals)
            
        # Convert the list of attention to a tensor
        attentionValues = torch.stack(attentionValues)
        
        # Reshape the tensor to a workable shape
        attentionValues = attentionValues.reshape((attentionValues.shape[1], attentionValues.shape[2], attentionValues.shape[3]*attentionValues.shape[0]))
        
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
    def __init__(self, inputDim=512, innerDim=2048, outputDim=512):
        super(FeedForward, self).__init__()
        
        # The fully connected model
        self.model = nn.Sequential(
            nn.Linear(inputDim, innerDim),
            nn.ReLU(),
            nn.Linear(innerDim, outputDim),
        ).to(device=device)
    
    
    
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
        self.FullyConnected = FeedForward(maxSentenceSize, 2048, maxSentenceSize).to(device=device)

        # Normalization layers
        self.norm1 = nn.LayerNorm((self.maxSentenceSize, inputEmbeddingSize)).to(device=device)
        self.norm2 = nn.LayerNorm((self.maxSentenceSize, inputEmbeddingSize)).to(device=device)
    
    
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
        att_add = att.float() + inputEncoding.float()
        att_norm = self.norm1(att_add.float())
        
        # Compute the forward value
        norm_res = att_norm.reshape(att_norm.shape[0], att_norm.shape[2], att_norm.shape[1])
        FC = self.FullyConnected(norm_res)
        FC = FC.reshape(FC.shape[0], FC.shape[2], FC.shape[1])
        
        # Send the fully connected output through an add and norm layer
        FC_add = FC + att_norm
        FC_norm = self.norm2(FC_add)
        
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
        
        # Create the first multi-head attention layer
        # with masking
        self.multiHead_Attention1 = multiHeadAttention(maxSentenceSize, inputVocabSize, outputVocabSize, inputEmbeddingSize, outputEmbeddingSize, attention_heads, keySize, querySize, valueSize, True)
        
        # Create the second multi-head attention layer
        self.multiHead_Attention2 = multiHeadAttention(maxSentenceSize, inputVocabSize, outputVocabSize, inputEmbeddingSize, outputEmbeddingSize, attention_heads, keySize, querySize, valueSize)
        
        # Create a fully connected layer
        self.FullyConnected = FeedForward(maxSentenceSize, 2048, maxSentenceSize).to(device=device)

        # Create three normalization layers
        self.norm1 = nn.LayerNorm((self.maxSentenceSize, outputEmbeddingSize)).to(device=device)
        self.norm2 = nn.LayerNorm((self.maxSentenceSize, outputEmbeddingSize)).to(device=device)
        self.norm3 = nn.LayerNorm((self.maxSentenceSize, outputEmbeddingSize)).to(device=device)
    
    
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
        att = self.multiHead_Attention1(outputEncoding)
        
        # Send the attiontion through an add and norm layer
        att_add = att.float() + outputEncoding.float()
        att_norm = self.norm1(att_add.float())
        
        # Compute the attiontion for the input results and output results
        att2 = self.multiHead_Attention2(inputRes, att_norm)
        
        # Send the new attiontion through an add and norm layer
        att2_add = att2 + outputEncoding
        att2_norm = self.norm2(att2_add.float())
        
        # Compute the forward value
        norm_res = att2_norm.reshape(att2_norm.shape[0], att2_norm.shape[2], att2_norm.shape[1])
        FC = self.FullyConnected(norm_res)
        FC = FC.reshape(FC.shape[0], FC.shape[2], FC.shape[1])
        
        # Send the fully connected output through an add and norm layer
        FC_add = FC + att2_norm
        FC_norm = self.norm3(FC_add)
        
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
    #   batchSize - Size of each minibatch
    #   stepsToSave - Number of steps until the model should be saved
    #   startStep - The starting step when training
    def __init__(self, maxSentenceSize, inputVocab, outputVocab, warmupSteps, inputEmbeddingSize, outputEmbeddingSize, attention_heads, keySize, querySize, valueSize, numBlocks, batchSize, stepsToSave, startStep=1):
        super(transformer, self).__init__()
        
        # Store the number of steps until the model should be saved
        self.stepsToSave = stepsToSave
        self.startStep = startStep
        
        # Store the maximum length
        self.maxSentenceSize = maxSentenceSize
        
        # Store the input and output vocabulary
        self.inputVocab = inputVocab
        self.outputVocab = outputVocab

	    # Save the number of warmup steps
        self.warmupSteps = warmupSteps
        
        # Store the input and output vocabulary, but inversed
        self.inputVocab_inv = {v: k for k, v in self.inputVocab.items()}
        self.outputVocab_inv = {v: k for k, v in self.outputVocab.items()}
        
        # Store the input and output vocabulary sizes
        self.inputVocabSize = len(inputVocab.keys())
        self.outputVocabSize = len(outputVocab.keys())
        
        # Store the input and output embedding sizes
        self.inputEmbeddingSize = inputEmbeddingSize
        self.outputEmbeddingSize = outputEmbeddingSize
        
        # Save the K, V, Q  sizes
        self.valueSize = valueSize
        self.keySize = keySize
        self.querySize = querySize
        
        # The word embedding layer for the inputs
        self.input_embedding_layer = nn.Embedding(self.inputVocabSize, inputEmbeddingSize).to(device=device)
        
        # The word embedding layer for the output
        self.output_embedding_layer = nn.Embedding(self.outputVocabSize, outputEmbeddingSize).to(device=device)
        
        # Store the number of blocks
        self.numBlocks = numBlocks
        
        # Create the input blocks
        self.inputBlocks = []
        for i in range(0, int(numBlocks/2)):
            self.inputBlocks.append(inputTransformerBlock(maxSentenceSize, self.inputVocabSize, self.outputVocabSize, inputEmbeddingSize, outputEmbeddingSize, attention_heads, keySize, querySize, valueSize))
            l = list(self.inputBlocks[-1].parameters())
            for b in range(0, len(l)):
                self.register_parameter(name="input"+str(i)+"_"+str(b), param=torch.nn.Parameter(l[b]))
            
        # Create the output blocks
        self.outputBlocks = []
        for i in range(0, int(numBlocks/2)):
            self.outputBlocks.append(outputTransformerBlock(maxSentenceSize, self.inputVocabSize, self.outputVocabSize, inputEmbeddingSize, outputEmbeddingSize, attention_heads, keySize, querySize, valueSize))
            l = list(self.outputBlocks[-1].parameters())
            for b in range(0, len(l)):
                self.register_parameter(name="output"+str(i)+"_"+str(b), param=torch.nn.Parameter(l[b]))
        
        
        # Output linear layer
        #self.finalLinear = nn.Sequential(nn.Linear(self.inputEmbeddingSize*self.maxSentenceSize, self.outputVocabSize)).to(device=device)
        self.finalLinear = nn.Sequential(nn.Linear(self.inputEmbeddingSize, self.outputVocabSize)).to(device=device)
        
        # The loss function for the transformer
        #self.loss_funct = torch.nn.CrossEntropyLoss().to(device=device)

        # Save the batch size for later use
        self.batchSize = batchSize
        
        # The optimizer for this model
        #self.optimizer = optim.Adam(list(self.parameters()), lr=alpha)
        self.optimizer = optim.Adam(sum(list(list(i.parameters()) for i in self.inputBlocks), []) + sum(list(list(j.parameters()) for j in self.outputBlocks), []) + list(self.finalLinear.parameters()) + list(self.input_embedding_layer.parameters()) + list(self.output_embedding_layer.parameters()))

    
    
    
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
                    words_num.append(self.inputVocabSize-1)
            words_num = torch.tensor(words_num, requires_grad=False, device=device)
            
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
            words_num = torch.tensor(words_num, requires_grad=False, device=device)
            
            # Embed the words
            embeddings = self.output_embedding_layer(words_num).detach()
        
        # Return the embeddings
        return embeddings

    
    # Embed a batch of input sentences to a batch of word embeddings
    # Inputs:
    #   batch - The batch of input sentences to embed
    def embedInputs(self, batch):
        # Send the words through the word embedding layer
        embeddings = []
        for sentence in batch:
            # Get the embedding
            embedding = self.embedWords(sentence, "Input")
            
            # pad the array with "PAD" values.
            embedding = torch.tensor(np.pad(embedding.cpu().numpy(), ((0, self.maxSentenceSize - embedding.shape[0]), (0, 0)), mode='constant', constant_values=self.inputVocab["<PAD>"]), requires_grad=True, device=device)
            
            # Get a poitional encoding vector which is the same
            # size as the sentence embedding
            posEnc = torch.tensor([[np.sin(i/np.power(10000, (2*embedding[i].shape[0])/self.inputEmbeddingSize))] if (i%2 == 0) else [np.cos(i/np.power(10000, (2*embedding[i].shape[0])/self.inputEmbeddingSize))] for i in range(0, embedding.shape[0])], requires_grad=True, device=device)
            
            # Apply positional encodings to the embedding
            embedding_enc = embedding + posEnc
            
            # Add the embedding to the embeddings array
            embeddings.append(embedding_enc)
        
        # Return the embeddings
        return torch.stack(embeddings)
    
    
    # Embed a batch of output sentences to a batch of word embeddings
    # Inputs:
    #   batch - The batch of output sentences to embed
    def embedOutputs(self, batch):
        # Send the words through the word embedding layer
        embeddings = []
        for sentence in batch:
            # Get the embedding
            embedding = self.embedWords(sentence, "Output")
            
            # pad the array with "PAD" values.
            embedding = torch.tensor(np.pad(embedding.cpu().numpy(), ((0, self.maxSentenceSize - embedding.shape[0]), (0, 0)), mode='constant', constant_values=self.outputVocab["<PAD>"]), requires_grad=True, device=device)
            
            # Get a poitional encoding vector which is the same
            # size as the sentence embedding
            posEnc = torch.tensor([[np.sin(i/np.power(10000, (2*embedding[i].shape[0])/self.outputEmbeddingSize))] if (i%2 == 0) else [np.cos(i/np.power(10000, (2*embedding[i].shape[0])/self.outputEmbeddingSize))] for i in range(0, embedding.shape[0])], requires_grad=True, device=device)
            
            # Apply positional encodings to the embedding
            embedding_enc = embedding + posEnc
            
            # Add the embedding to the embeddings array
            embeddings.append(embedding_enc)
        
        # Return the embeddings
        return torch.stack(embeddings)

    
    
    # Embed a batch of output sentences to a batch of word embeddings
    # Inputs:
    #   batch - The batch of output sentences to embed
    def embedOutputs_OnlyPosEnc(self, batch):
        # Send the words through the word embedding layer
        embeddings = []
        for sentence in batch:
            # Get the embedding
            embedding = sentence
            #embedding = self.embedWords(sentence, "Output")
            
            # pad the array with "PAD" values.
            #paddedEmbedding = [t for t in embedding[:currentIndex]]
            paddedEmbedding = [t for t in embedding]
            for i in range(0, self.maxSentenceSize - embedding.shape[0]):
                paddedEmbedding.append(self.embedWords(["<PAD>"], "Output")[0])
            paddedEmbedding = torch.stack(paddedEmbedding)
            
            # Get a poitional encoding vector which is the same
            # size as the sentence embedding
            idx = paddedEmbedding.shape[0]-1
            #posEnc =  torch.tensor([np.sin(idx/np.power(10000, (2*paddedEmbedding[idx].shape[0])/self.outputEmbeddingSize))] if (idx%2 == 0) else [np.cos(idx/np.power(10000, (2*paddedEmbedding[idx].shape[0])/self.outputEmbeddingSize))], dtype=torch.float, requires_grad=True, device=device)
            posEnc = torch.tensor([[np.sin(i/np.power(10000, (2*paddedEmbedding[i].shape[0])/self.outputEmbeddingSize))] if (i%2 == 0) else [np.cos(i/np.power(10000, (2*paddedEmbedding[i].shape[0])/self.outputEmbeddingSize))] for i in range(0, paddedEmbedding.shape[0])], requires_grad=True, device=device)
            #posEnc =  torch.tensor([[0] if i != paddedEmbedding.shape[0]-1 else [np.sin(idx/np.power(10000, (2*paddedEmbedding[idx].shape[0])/self.outputEmbeddingSize))] if (idx%2 == 0) else [np.cos(idx/np.power(10000, (2*paddedEmbedding[idx].shape[0])/self.outputEmbeddingSize))] for i in range(0, paddedEmbedding.shape[0])], dtype=torch.float, requires_grad=True, device=device)
            #posEnc = torch.tensor([[0] if i == embedding.shape[0]-1 else [np.sin(i/np.power(10000, (2*embedding[i].shape[0])/self.outputEmbeddingSize))] if (i%2 == 0) else [np.cos(i/np.power(10000, (2*embedding[i].shape[0])/self.outputEmbeddingSize))] for i in range(0, embedding.shape[0])], dtype=torch.float, requires_grad=True, device=device)
            
            # Apply positional encodings to the embedding
            embedding_enc = paddedEmbedding + posEnc
            
            # Add the embedding to the embeddings array
            embeddings.append(embedding_enc)
        
        # Return the embeddings
        return torch.stack(embeddings)
    
    
    # Embed a batch of output sentences to a batch of word embeddings
    # without position encodings
    # Inputs:
    #   batch - The batch of output sentences to embed
    def embedOutputs_NoPosEnc(self, batch):
        # Send the words through the word embedding layer
        embeddings = []
        for sentence in batch:
            # The sentence embedding
            embedding = []
            
            # Iterate over all words in the batch
            for word in range(0, self.maxSentenceSize):
                # If the word exists, encode the word
                if word < len(sentence):
                    embedding.append(self.outputVocab[sentence[word]])
                
                # If the word does not exist, encode the word as a padding character
                else:
                    embedding.append(self.outputVocabSize-1)
            
            # Add the embedding to the embeddings
            embeddings.append(torch.tensor(embedding, requires_grad=False, device=device))
        
        return torch.stack(embeddings)

    
    
    # The Cross entropy loss function
    # Inputs:
    #   p - The probabilities we want (Probably a one-hot vector)
    #   q - The probabilities the model predicted
    def CrossEntropyLoss(self, p, q):
        q = torch.where(q == torch.tensor(0.000001, dtype=torch.float32), q)
        q = torch.where(q == torch.tensor(0.999999, dtype=torch.float32), q)
        return -torch.sum(p*torch.log(q) + (1-p)*torch.log(1-q), dim=-1)
    
    
    
    
    # Train the model
    # Input:
    #   x - The batch of sentences to translate
    #   Y - The batch of sentences to translate to
    #   numSteps - Number of steps to train the model
    #   modelSaveName - The name of the file to save the model
    #   clipVal - The bound used to clip the gradients
    def trainModel(self, x, Y, numSteps, modelSaveName, clipVal):
        # Split the data into batches
        slices = [self.batchSize*i for i in range(1, int(len(x)/self.batchSize)+1)]+[(int(len(x)/self.batchSize)*self.batchSize)+len(x)%self.batchSize]
        x_batches = np.split(np.array(x), slices)[:-1]
        Y_batches = np.split(np.array(Y), slices)[:-1]
        
        # Arrays of data used to graph
        losses = []
        stepCounts = []

        # Iterate and update the model
        for iter in range(self.startStep, numSteps+1):
            # Update the learning rate
            alpha = (self.valueSize**-0.5)*min((iter**-0.5), (iter*(self.warmupSteps**-1.5)))
            for g in self.optimizer.param_groups:
                g["lr"] = alpha

            # Total loss of all batches
            totalLoss = 0
            
            # The lowest loss so far
            lowestLoss = np.inf

            # Iterate over all batches
            for batch_num in range(0, len(x_batches)):
                # Store the batch data
                x_sub = x_batches[batch_num]
                Y_sub = Y_batches[batch_num]
                
                # If the batch is empty, skip it
                if x_sub.shape[0] == 0:
                    continue

                # Send the words through the word embedding layer
                in_embeddings = self.embedInputs(x_sub)
                
                # Send the embeddings through the input layer
                inputRes = in_embeddings
                for i in range(0, int(self.numBlocks/2)):
                    # Send the inputs through the input block
                    inputRes = self.inputBlocks[i](inputRes)
                    
                    
                    
                    
                # Create the initial sentences to get output from
                output_init = []
                for i in range(0, len(Y_sub)):
                    output_init.append(torch.tensor([], device=device, requires_grad=True))

                
                newWords = ["<START>" for i in range(0, slices[batch_num])]
                wordIndex = 1
                
                
                # Store the softmax values for each word
                softVals = torch.tensor([], device=device)
                
                # Store the current output sentences which will
                # update as the model predicts new outputs
                outputMatrix = torch.stack(output_init)
                
                    
                # While the max sentence length has not been reached, 
                # create the new sentence
                while (wordIndex < self.maxSentenceSize):
                    # Add the new word to the arrays
                    newOutputMatrix = []
                    for i in range(0, len(Y_sub)):
                        newOutputMatrix.append(torch.cat((outputMatrix[i][:wordIndex-1], torch.tensor(self.embedWords([newWords[i]], "Output"), device=device, requires_grad=True, dtype=torch.float))))
                        #newOutputMatrix.append(torch.tensor(self.embedWords([newWords[i]], "Output"), device=device, requires_grad=True, dtype=torch.float))
                    outputMatrix = torch.stack(newOutputMatrix)
                    
                    # Send the outputs through the word embedding layer
                    outputMatrix_posEnc = self.embedOutputs_OnlyPosEnc(outputMatrix)
                    
                    # Send the output through the output blocks
                    for block in range(0, int(self.numBlocks/2)):
                        outputPreds = self.outputBlocks[block](inputRes, outputMatrix_posEnc)
                    
                    # Reshape the output matrix so that the linear layer
                    # converts the sentence length and word embedding dimesnions
                    # to a single output dictionary size dimension
                    #outputPredsReshaped = outputPreds.reshape((outputPreds.shape[0], outputPreds.shape[1]*outputPreds.shape[2]))
                        
                    # Send the output through the linear layer where
                    # the linear layer has the same number of nodes
                    # as the output Vocab
                    linear = self.finalLinear(outputPreds)
                    #linear[:, -1] = 0 # Don't allow <PAD> to be predicted
                    softmax = nn.Softmax(dim=-1)(linear)
                    
                    # Add the max softmax values to the softVals array
                    softmax_sub = softmax[:, wordIndex]
                    #softmax_sub = softmax
                    softmax_sub = softmax_sub.reshape(softmax_sub.shape[0], 1, softmax_sub.shape[1])
                    softVals = torch.cat((softVals, softmax_sub), dim=-2)
                    
                    # Get the indices of the max softmax values
                    dictVals = torch.argmax(softmax, dim=-1)
                    
                    # Get the new word indices
                    wordIdx = dictVals[:, wordIndex]
                    #wordIdx = dictVals
                    
                    # Get the new words
                    newWords = []
                    for i in range(0, wordIdx.shape[0]):
                        newWords.append(self.outputVocab_inv[wordIdx[i].item()])
                        
                    # Increase the word index
                    wordIndex += 1
                

                # Add the final words to the output matrix
                newOutputMatrix = []
                for i in range(0, len(Y_sub)):
                    newOutputMatrix.append(torch.cat((outputMatrix[i][:wordIndex-1], torch.tensor(self.embedWords([newWords[i]], "Output"), device=device, requires_grad=True, dtype=torch.float))))
                    #newOutputMatrix.append(torch.tensor(self.embedWords([newWords[i]], "Output"), device=device, requires_grad=True, dtype=torch.float))
                outputMatrix = torch.stack(newOutputMatrix)
                
                
                # Send the outputs through the word embedding layer without
                # positional encodings
                out_embeddings_NoPosEnc = self.embedOutputs_NoPosEnc(Y_sub)
                
                # Get the indices of the max softmax values
                dictVals = torch.argmax(softVals, dim=-1)
                
                # Get the decoded new words
                output = []
                for i in range(0, x_sub.shape[0]):
                    #output.append(self.outputVocab_inv[dictVals[i].cpu().numpy().item()])
                    output.append([self.outputVocab_inv[j.cpu().numpy().item()] for j in dictVals[i]])
                    # print(x[i][:10])
                    # print(output[i][:10])
                    # print()
                
                # One hot encode the output embeddings
                out_embeddings_NoPosEnc_oneHot = nn.functional.one_hot(out_embeddings_NoPosEnc, num_classes=softmax.shape[-1])
                
                # Get the indices of the last word in each sentence
                lastWordIdx = [len(Y_sub[i]) for i in range(0, len(Y_sub))]

                # Get the loss
                loss = []
                for i in range(0, in_embeddings.shape[0]):
                    # Split the embeddings and softmax values by the last word indices
                    # so the loss doesn't count the <PAD> terms
                    out_embeddings_NoPosEnc_oneHot_sub = out_embeddings_NoPosEnc_oneHot[i][:lastWordIdx[i]]
                    softmax_sub = softVals[i][:lastWordIdx[i]]

                    loss.append(self.CrossEntropyLoss(out_embeddings_NoPosEnc_oneHot_sub, softmax_sub).mean())
                loss = torch.stack(loss)
                
                # sum the loss
                s = loss.sum()
                totalLoss += s.detach().cpu().numpy().item()

                if torch.isnan(s):
                    print()
                
                # Update the gradients
                s.backward(retain_graph=False)
                
                # Step the optimizer
                self.optimizer.step()

                # Clip the gradients
                torch.nn.utils.clip_grad_norm_(sum(list(list(i.parameters()) for i in self.inputBlocks), []) + sum(list(list(j.parameters()) for j in self.outputBlocks), []) + list(self.finalLinear.parameters()) + list(self.input_embedding_layer.parameters()) + list(self.output_embedding_layer.parameters()), clipVal)
                
                # Zero the optimizers
                self.optimizer.zero_grad()
            
            
            # Save the model if the number of steps has
            # passed and the loss is less than the
            # lowest loss
            if iter%self.stepsToSave == 0:
                # Add the total loss to the losses array
                losses.append(totalLoss)
                stepCounts.append(iter)
                
                if totalLoss < lowestLoss:
                    # Save the model
                    self.saveModel(modelSaveName)
                    lowestLoss = totalLoss
            
            
            # Show the total batch loss
            print(f"Step #{iter}")
            print(f"English: {x[-1][:10]}")
            print(f"Spanish: {Y[-1][:10]}")
            print(f"Spanish Prediction: {output[-1][:10]}")
            print()
            if (len(output) > 1):
                print(f"English: {x[-2][:10]}")
                print(f"Spanish: {Y[-2][:10]}")
                print(f"Spanish Prediction: {output[-2][:10]}")
            print(f"Total batch loss: {totalLoss}")
            print()
        
        # Return the data used for graphing
        return losses, stepCounts
        
    
    
    # Translate the given sentence(s)
    # Input:
    #   x - The batch of sentences to translate
    def forward(self, x):
        # Breakup the sentence(s)
        inputs = [re.sub(r'[^\w\s]', '', i.replace("\xa0", " ")).lower().split(" ") for i in x]

        # Add <START> and <END> stop words to the sentence
        inputs = addSTARTAndEND(inputs)

        # Send the sentences through the word embedding layer
        in_embeddings = self.embedInputs(inputs)
        
        # Send the embeddings through the input layer
        inputRes = in_embeddings
        for i in range(0, int(self.numBlocks/2)):
            # Send the inputs through the input block
            inputRes = self.inputBlocks[i](inputRes)
            
            
            
            
        # Create the initial sentences to get output from
        output_init = []
        for i in range(0, len(inputs)):
            output_init.append(torch.tensor([], device=device, requires_grad=True))

        
        newWords = ["<START>" for i in range(0, len(inputs))]
        wordIndex = 1
        
        
        # Store the softmax values for each word
        softVals = torch.tensor([], device=device)
        
        # Store the current output sentences which will
        # update as the model predicts new outputs
        outputMatrix = torch.stack(output_init)
        
        # Store the output in word form
        wordOutputMatrix = [[] for i in range(0, len(inputs))]
        
            
        # While the max sentence length has not been reached, 
        # create the new sentence
        while (wordIndex < self.maxSentenceSize):
            # Add the new word to the arrays
            newOutputMatrix = []
            for i in range(0, len(inputs)):
                newOutputMatrix.append(torch.cat((outputMatrix[i][:wordIndex-1], torch.tensor(self.embedWords([newWords[i]], "Output"), device=device, requires_grad=True, dtype=torch.float))))
                #newOutputMatrix.append(torch.tensor(self.embedWords([newWords[i]], "Output"), device=device, requires_grad=True, dtype=torch.float))
            outputMatrix = torch.stack(newOutputMatrix)
            
            # Add the words to the saved output
            for i in range(0, len(inputs)):
                wordOutputMatrix[i].append(newWords[i])
            
            # Send the outputs through the word embedding layer
            outputMatrix_posEnc = self.embedOutputs_OnlyPosEnc(outputMatrix)
            
            # Send the output through the output blocks
            for block in range(0, int(self.numBlocks/2)):
                outputPreds = self.outputBlocks[block](inputRes, outputMatrix_posEnc)
            
            # Reshape the output matrix so that the linear layer
            # converts the sentence length and word embedding dimesnions
            # to a single output dictionary size dimension
            #outputPredsReshaped = outputPreds.reshape((outputPreds.shape[0], outputPreds.shape[1]*outputPreds.shape[2]))
                
            # Send the output through the linear layer where
            # the linear layer has the same number of nodes
            # as the output Vocab
            linear = self.finalLinear(outputPreds)
            #linear[:, -1] = 0 # Don't allow <PAD> to be predicted
            softmax = nn.Softmax(dim=-1)(linear)
            
            # Add the max softmax values to the softVals array
            softmax_sub = softmax[:, wordIndex]
            #softmax_sub = softmax
            softmax_sub = softmax_sub.reshape(softmax_sub.shape[0], 1, softmax_sub.shape[1])
            softVals = torch.cat((softVals, softmax_sub), dim=-2)
            
            # Get the indices of the max softmax values
            dictVals = torch.argmax(softmax, dim=-1)
            
            # Get the new word indices
            wordIdx = dictVals[:, wordIndex]
            #wordIdx = dictVals
            
            # Get the new words
            newWords = []
            for i in range(0, wordIdx.shape[0]):
                newWords.append(self.outputVocab_inv[wordIdx[i].item()])    
            
            # Increase the word index
            wordIndex += 1
        

        # Add the final words to the output matrix
        for i in range(0, len(inputs)):
                wordOutputMatrix[i].append(newWords[i])
        
        # Return the decoded sentences
        return wordOutputMatrix



    # Save a model to the specified path name
    # Input:
    #   fileName - The name of the file to save the model to
    def saveModel(self, fileName):
        # Get the last separator in the filename
        dirName = "/".join(fileName.split("/")[0:-1])
    
        # If the directory doesn't exist, create it
        try:
            if (not os.path.isdir(dirName) and dirName != ''):
                os.makedirs(dirName, exist_ok=True)
        
            torch.save(self.state_dict(), fileName)
        
        except:
            torch.save(self.state_dict(), fileName)
            
    
    
    # Load a model from the specified path name
    # Input:
    #   fileName - The name of the file to load the model from
    def loadModel(self, fileName):
        # If the file doesn't exist, raise an error
        if (not os.path.isfile(fileName)):
            raise Exception("Specified model file does no exist")
        
        # Load the model
        self.load_state_dict(torch.load(fileName))
        self.eval()