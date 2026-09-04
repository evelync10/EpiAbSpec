#%%
from text import Tokenizer
import pandas as pd
import torch
from torch.nn.utils.rnn import pad_sequence
import torch.nn.functional as F
import pickle

# Data Preprocessing:
# Read the CSV file into a DataFrame
data = pd.read_csv('./vj/VJgenenew.csv', sep=',')

# Preprocess species and gene columns, and create species-gene combinations
data['H_Species'] = data['H_Species'].str.replace(' ', '')
data['L_Species'] = data['L_Species'].str.replace(' ', '')
data['species_H_Vgene'] = data['H_Species'] + '-' + data['H_Vgene']
data['species_H_Jgene'] = data['H_Species'] + '-' + data['H_Jgene']
data['species_L_Vgene'] = data['L_Species'] + '-' + data['L_Vgene']
data['species_L_Jgene'] = data['L_Species'] + '-' + data['L_Jgene']

# Extract unique gene sequences for heavy and light chains
vhgene = data['species_H_Vgene'].drop_duplicates().values.tolist()
vlgene = data['species_L_Vgene'].drop_duplicates().values.tolist()
v_gene = vhgene + vlgene

jhgene = data['species_H_Jgene'].drop_duplicates().values.tolist()
jlgene = data['species_L_Jgene'].drop_duplicates().values.tolist()
j_gene = jhgene + jlgene
v_gene = [str(gene) for gene in v_gene]
j_gene = [str(gene) for gene in j_gene]

testdata = pd.read_csv('./vj/test_VJgenenew.csv', sep=',')

# Preprocess species and gene columns, and create species-gene combinations
testdata['H_Species'] = testdata['H_Species'].str.replace(' ', '')
testdata['L_Species'] = testdata['L_Species'].str.replace(' ', '')
testdata['species_H_Vgene'] = testdata['H_Species'] + '-' + testdata['H_Vgene']
testdata['species_H_Jgene'] = testdata['H_Species'] + '-' + testdata['H_Jgene']
testdata['species_L_Vgene'] = testdata['L_Species'] + '-' + testdata['L_Vgene']
testdata['species_L_Jgene'] = testdata['L_Species'] + '-' + testdata['L_Jgene']

# Extract unique gene sequences for heavy and light chains
testvhgene = testdata['species_H_Vgene'].drop_duplicates().values.tolist()
testvlgene = testdata['species_L_Vgene'].drop_duplicates().values.tolist()
testv_gene = testvhgene + testvlgene

testjhgene = testdata['species_H_Jgene'].drop_duplicates().values.tolist()
testjlgene = testdata['species_L_Jgene'].drop_duplicates().values.tolist()
testj_gene = testjhgene + testjlgene
testv_gene = [str(gene) for gene in testv_gene]
testj_gene = [str(gene) for gene in testj_gene]



# Tokenization:
# Initialize and fit the Tokenizer on gene sequences

tokenizer = Tokenizer(lower=False)
tokenizer.fit_on_texts(v_gene)
tokenizer.fit_on_texts(j_gene)
tokenizer.fit_on_texts(testv_gene)
tokenizer.fit_on_texts(testj_gene)
with open('VJ_tokenizer.pkl', 'wb') as f:
    pickle.dump(tokenizer, f)
#%%
print(data)
print(tokenizer.word_index)
print(tokenizer.texts_to_sequences(testv_gene))
#%%
# Load the saved tokenizer configuration

# Process each unique identifier
ids = testdata['Name Prefix'].drop_duplicates().values.tolist()
H_Vgene = None
H_Jgene = None
L_Vgene = None
L_Jgene = None
max_sequence_length = 0
# Iterate through identifiers
for id in ids:
    # Select rows based on identifier
    selected_row = testdata[testdata['Name Prefix'] == id]
    H_Vgene = str(selected_row['species_H_Vgene'].values[0])
    H_Jgene = str(selected_row['species_H_Jgene'].values[0])
    L_Vgene = str(selected_row['species_L_Vgene'].values[0])
    L_Jgene = str(selected_row['species_L_Jgene'].values[0])
    # print(H_Vgene, H_Jgene, L_Vgene, L_Jgene)
    # Tokenize and pad gene sequences
    tokenized_sequence = tokenizer.texts_to_sequences([H_Vgene, H_Jgene, L_Vgene, L_Jgene])
    max_sequence_length = max(max_sequence_length, max(len(seq) for seq in tokenized_sequence))
    # print(tokenized_sequence)
    # print(max_sequence_length)
    # if max_sequence_length !=5:
    #     print(max_sequence_length)
    max_sequence_length = 5
    padded_sequence = pad_sequence([torch.tensor(seq + [0] * (max_sequence_length - len(seq))) for seq in tokenized_sequence],
                                   batch_first=True, padding_value=0)
    print(padded_sequence)
    if padded_sequence.view(1,-1).shape != torch.Size([1, 20]):
        print(id)
        print(padded_sequence.view(1,-1).shape)

    # padded_sequence = ([torch.tensor(seq) for seq in tokenized_sequence], batch_first=True, padding_value=0)
    # Save the padded sequence to a file
    torch.save(padded_sequence.view(1,-1), './VJsequence/AAI/' + f'{id}' + '_paratope.tensor')

#%%
import torch
import torch.nn as nn

class VDJGeneEncoder(nn.Module):
    def __init__(self, vocab_size, v_embed, j_embed,dropout_rate):
        super(VDJGeneEncoder, self).__init__()

        self.v_embedding = nn.Embedding(vocab_size,v_embed)
        self.j_embedding = nn.Embedding(vocab_size, j_embed)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        h_v_embedding = self.v_embedding(x[:, 1:6])
        h_j_embedding = self.j_embedding(x[:, 1, :])
        l_v_embedding = self.v_embedding(x[:, 2, :])
        l_j_embedding = self.j_embedding(x[:, 3, :])
        print(h_v_embedding.shape)
        print(h_j_embedding.shape)

        # Flatten the embeddings
        h_v_flat = h_v_embedding.view(x.size(0), -1)
        h_j_flat = h_j_embedding.view(x.size(0), -1)
        l_v_flat = l_v_embedding.view(x.size(0), -1)
        l_j_flat = l_j_embedding.view(x.size(0), -1)
        print(h_v_flat.shape)
        print(h_j_flat.shape)

        # Concatenate the flattened embeddings
        concatenated = torch.cat([h_v_flat, h_j_flat, l_v_flat, l_j_flat], dim=1)
        print(concatenated.shape)

        # Apply dropout
        output = self.dropout(concatenated)
        return output

vocab_size = 238  # Adjust based on your tokenizer
v_embed_dim = 16
j_embed_dim = 8
dropout_rate = 0.2

vj_gene_encoder = VDJGeneEncoder(vocab_size, v_embed_dim, j_embed_dim, dropout_rate)
print(vj_gene_encoder)
#%%
tensor1 = torch.tensor([[  2,   6,  41,  19,   0],
        [  2,  42,   1,   0,   0],
        [  2,  26, 107,   1,   0],
        [  2, 100,   1,   0,   0]])

tensor2 = torch.tensor([[  2,   5,  30,  32,   0],
        [  2,  48,   4,   0,   0],
        [  2,  40,  82,   1,   0],
        [  2, 100,   1,   0,   0]])

# Remove the extra comma
combined_batch = torch.stack([tensor1, tensor2], dim=0)
# print(combined_batch)
output = vj_gene_encoder(combined_batch)
print(output)
#%%
ts = torch.load('/media/dulab/data1/GraphEAI_final_epiab/Feature/VJsequence/AAI/1a2y_1_paratope.tensor')
print(ts)