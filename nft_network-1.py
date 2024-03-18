#%%
"""
Retrieve NFT transactions 
Retrieve Wallets that have more than n unique transactions 
Look at transactions between collections 

"""


from web3 import Web3
from datetime import datetime
import pandas as pd
import numpy as np 
import json
import time
import glob 
import regex as re
from alchemy_sdk_py import Alchemy
from sklearn.preprocessing import OrdinalEncoder
import networkx as nx
import matplotlib.pyplot as plt

nft_collections = pd.read_csv('export-nft-top-contracts-1710592633736.csv')
nft_collections


# Connect to Alchemy
API = 'CeaJwmD8u0qcSnGwuKPwbN_jGryIlD61'
alchemy = Alchemy(api_key=API)

accounts = nft_collections['Token']


all_transactions = {}

for account in accounts[0:50]:
    transfers, page_key = alchemy.get_asset_transfers(from_address=account, max_count=1000)
    all_transactions[account] = pd.DataFrame(transfers)

transactions = pd.concat(all_transactions)
transactions = transactions.reset_index(0)

df = transactions.merge(nft_collections, left_on='level_0', right_on='Token')[['Token Name', 'asset', 'from', 'to', 'tokenId', 'blockNum']]

df['tokenId'] =  OrdinalEncoder().fit_transform(df['tokenId'].values.reshape(-1, 1))
#%%
base = df.copy(deep=True)
#%%
df = base.copy(deep=True)
popular_token = df.drop_duplicates()['Token Name'].value_counts().sort_values().tail(15).index.to_list()
large_buyers = df['to'].value_counts().sort_values().tail(100).index.to_list()

df = df.loc[df['Token Name'].isin(popular_token)].drop_duplicates()
#%%
df.drop_duplicates()#.groupby(['from', 'to', 'Token Name']).count()
#%%
import math

# Create the graph
G = nx.Graph()

min_degree_threshold = 3

# Add tokenId and address nodes, including Token Name as an attribute for tokenId nodes
tokenId_set = set(df['tokenId'].unique())
all_addresses = set(df['from'].unique()).union(set(df['to'].unique()))
token_name_attr = df.set_index('tokenId')['Token Name'].to_dict()
for tokenId in tokenId_set:
    G.add_node(tokenId, label=str(tokenId), type='token', token_name=token_name_attr[tokenId])
for address in all_addresses:
    G.add_node(address, label='', type='address')

# Add edges based on transactions
for index, row in df.iterrows():
    G.add_edge(row['from'], row['tokenId'])
    G.add_edge(row['tokenId'], row['to'])


# Node sizes based on degree, with different scales for tokenId and address nodes
node_degrees = G.degree()
base_size = 20
scale_token = 20
scale_address = 20  # Increased scale for address nodes to visually differentiate from tokenIds
node_sizes = [(node_degrees[node] * scale_token + base_size if G.nodes[node]['type'] == 'token' else node_degrees[node] * scale_address + base_size) for node in G.nodes()]




# Manually setting node positions
pos = {}
outer_circle_radius = 1.0
address_angle = 2 * math.pi / len(all_addresses)
for i, address in enumerate(all_addresses):
    theta = i * address_angle
    pos[address] = (outer_circle_radius * math.cos(theta), outer_circle_radius * math.sin(theta))

inner_circle_radius = 0.5
token_angle = 2 * math.pi / len(tokenId_set)
for i, tokenId in enumerate(tokenId_set):
    theta = i * token_angle
    pos[tokenId] = (inner_circle_radius * math.cos(theta), inner_circle_radius * math.sin(theta))

# Assign colors to tokenId nodes based on their Token Name
unique_token_names = df['Token Name'].unique()
color_map = {token: plt.cm.tab10(i / len(unique_token_names)) for i, token in enumerate(unique_token_names)}
node_colors = [color_map[G.nodes[node]['token_name']] if 'token_name' in G.nodes[node] else 'skyblue' for node in G.nodes()]


G_filtered = nx.Graph()


# Add only those nodes and edges to G_filtered that meet the degree threshold
for node, degree in dict(G.degree()).items():
    if degree >= min_degree_threshold:
        G_filtered.add_node(node, **G.nodes[node])

for node1, node2, data in G.edges(data=True):
    if G.degree(node1) >= min_degree_threshold and G.degree(node2) >= min_degree_threshold:
        G_filtered.add_edge(node1, node2, **data)

# Now, recalculating node positions, sizes, and colors for G_filtered
pos_filtered = {}  # Position dictionary for G_filtered
node_sizes_filtered = []  # Node sizes for G_filtered
node_colors_filtered = []  # Node colors for G_filtered



# Calculate positions, sizes, and colors for the filtered nodes
for node in G_filtered.nodes:
    pos_filtered[node] = pos[node]  # Reuse the original positions
    node_sizes_filtered.append(G_filtered.degree(node) * scale_token + base_size if G_filtered.nodes[node]['type'] == 'token' else G_filtered.degree(node) * scale_address + base_size)
    node_colors_filtered.append(color_map[G_filtered.nodes[node]['token_name']] if 'token_name' in G_filtered.nodes[node] else 'skyblue')

# Draw the filtered graph
plt.figure(figsize=(14, 12))
nx.draw_networkx_nodes(G_filtered, pos_filtered, node_size=node_sizes_filtered, node_color=node_colors_filtered, alpha=0.7)
nx.draw_networkx_edges(G_filtered, pos_filtered, edge_color=node_colors_filtered, alpha=0.1, width=1)

# Since labels might make the graph cluttered, consider commenting out the label drawing line if not needed
# nx.draw_networkx_labels(G_filtered, pos_filtered, font_size=12, font_weight='bold')

# Legend for Token Names and Address Nodes, adapted for the filtered graph
for token_name, color in color_map.items():
    plt.plot([0], [0], color=color, label=token_name, linestyle='None', marker='o')
plt.plot([0], [0], color='skyblue', label='Address Nodes', linestyle='None', marker='o')
plt.legend(title="NFT Collections", loc='lower right')

plt.title('Blockchain NFT Transaction Network')
plt.axis('off')
plt.show()
# %%
