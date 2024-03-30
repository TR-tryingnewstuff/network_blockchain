#%%
from web3 import Web3
from datetime import datetime
import pandas as pd
import numpy as np 
import json
import time
import glob 
import regex as re
from alchemy_sdk_py import Alchemy
import matplotlib.pyplot as plt
import networkx as nx

#%%

def get_accounts_name_df():

    files = glob.glob('accounts_eth/*.csv')

    accounts = []
    for f in files:

        accounts.append(pd.read_csv(f))

    df = pd.concat(accounts).dropna()

    return df 

wallets = get_accounts_name_df()

print(wallets['Name Tag'].str.contains(':').sum())

wallets['Name Tag'] = [re.sub('[^a-zA-Z :]', '', tag) for tag in wallets['Name Tag']] # ? Remove numbers from tag names
wallets = wallets.loc[np.invert(wallets['Name Tag'].str.contains('Fee Recipient'))] # ? Remove "Fee Recipient" Adress 
wallets['Name Tag'] = wallets['Name Tag'].apply(str.strip)  # ? Remove Leading / Trailing Whitespaces

def get_first_word(word):

    return word.split(':')[0]

wallets['Name Tag'] =  wallets['Name Tag'].apply(get_first_word)

wallets = wallets.set_index('Name Tag')


#%%

# Connect to Alchemy
API = 'CeaJwmD8u0qcSnGwuKPwbN_jGryIlD61'
alchemy = Alchemy(api_key=API)

accounts = wallets['Address']


all_transactions = {}

for account in accounts[0:50]:
    #! only get transactions between large wallets to test 
    transfers, page_key = alchemy.get_asset_transfers(from_address=account, max_count=100)
    all_transactions[account] = pd.DataFrame(transfers)

transactions = pd.concat(all_transactions)


#%%
transactions = pd.read_csv('eth_network_df.csv')

df = transactions.reset_index().merge(wallets['Address'].reset_index(), left_on='Unnamed: 0', right_on='Address')
df = df.drop(['Unnamed: 0', 'Unnamed: 1', 'erc721TokenId', 'erc1155Metadata', 'blockNum', 'uniqueId', 'hash', 'rawContract'], axis=1)


mapping_dict = df[['Name Tag', 'Address']].drop_duplicates().set_index('Address').to_dict()['Name Tag']

df['from'] =  df['from'].apply(lambda x : mapping_dict.get(x, None) if mapping_dict.get(x, None) != None else x)
df['to'] = df['to'].apply(lambda x : mapping_dict.get(x, None) if mapping_dict.get(x, None) != None else x)
df

df = df.loc[df['from'].apply(type) != type(None)]
df = df.loc[df['to'].apply(type) != type(None)]

print(len(df))

def is_informative(tag):
    
    return re.sub('0x', '', tag) == tag


G = nx.from_pandas_edgelist(df, 'from', 'to', edge_attr='value', edge_key='Name Tag')

informative_labels = {row['Name Tag']: row['Name Tag'] for index, row in df.iterrows() if is_informative(row['Name Tag'])}

# Node sizes based on degree
node_sizes = [G.degree(n) * 100 for n in G.nodes()]

# Adjusting node colors based on 'Name Tag'
node_colors = ['skyblue' if node in informative_labels else 'orange' for node in G.nodes()]

# Edge widths normalization
values = nx.get_edge_attributes(G, 'value')
max_value = max(values.values()) if values else 1

# Layout for positioning
pos = nx.kamada_kawai_layout(G)

# Drawing the network
plt.figure(figsize=(15, 12))
nx.draw(G, pos, with_labels=True, labels=informative_labels, node_size=node_sizes, node_color=node_colors, edge_color='lightgray', font_size=8, font_weight='bold')

plt.title('Blockchain Transaction Network')
plt.axis('off')
plt.show()

#%%
