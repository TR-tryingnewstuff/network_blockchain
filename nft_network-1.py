
#%%
"""
Retrieve NFT transactions 
Retrieve Wallets that have more than n unique transactions 
Look at transactions between collections 

"""

from web3 import Web3
import pandas as pd
from alchemy_sdk_py import Alchemy
from sklearn.preprocessing import OrdinalEncoder
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
#%%
nft_collections = pd.read_csv('export-nft-top-contracts-1710592633736.csv')
nft_collections


# Connect to Alchemy
API = 'CeaJwmD8u0qcSnGwuKPwbN_jGryIlD61'
alchemy = Alchemy(api_key=API)

accounts = nft_collections['Token']
accounts
#%% 
all_transactions = {}

for account in accounts[0:]:
    transfers, page_key = alchemy.get_asset_transfers(from_address=account, max_count=1000)
    all_transactions[account] = pd.DataFrame(transfers)

transactions = pd.concat(all_transactions)
transactions.to_csv('nft_transactions_df.csv')

#%%
transactions = pd.read_csv('nft_transactions_df.csv') #transactions.reset_index(0)

df = transactions.merge(nft_collections, left_on='Unnamed: 0', right_on='Token')[['Token Name', 'asset', 'from', 'to', 'tokenId', 'blockNum']]

df['tokenId'] =  OrdinalEncoder().fit_transform(df['Token Name'].values.reshape(-1, 1))


collection_with_multiple_owners = (df.groupby('Token Name').count() > 50).replace(False, np.nan).dropna().index.to_list()
df = df.loc[df['Token Name'].isin(collection_with_multiple_owners)]#.sample(frac=0.2)


n_wallets_with_highest_n_transactions = df.groupby('to').count().sort_values('asset').tail(1500).index.to_list()
df = df.loc[df['to'].isin(n_wallets_with_highest_n_transactions)]


#%%
import math
def plot_network_nft():
    # Create the graph
    G = nx.Graph()

    min_degree_threshold = 1

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

    # Assign colors to tokenId nodes based on their Token Name
    unique_token_names = df['Token Name'].unique()
    color_map = {token: plt.cm.rainbow(i / len(unique_token_names)) for i, token in enumerate(unique_token_names)}


    G_filtered = nx.Graph()

    # Add only those nodes and edges to G_filtered that meet the degree threshold
    for node, degree in dict(G.degree()).items():
        if degree >= min_degree_threshold:
            G_filtered.add_node(node, **G.nodes[node])

    for node1, node2, data in G.edges(data=True):
        if G.degree(node1) >= min_degree_threshold and G.degree(node2) >= min_degree_threshold:
            G_filtered.add_edge(node1, node2, **data)

    # Now, recalculating node positions, sizes, and colors for G_filtered

    node_sizes_filtered = []  # Node sizes for G_filtered
    node_colors_filtered = []  # Node colors for G_filtered
    edge_sizes_filtered = []


    # Calculate positions, sizes, and colors for the filtered nodes
    for node in G_filtered.nodes:
        node_sizes_filtered.append(G_filtered.degree(node) * scale_token + base_size if G_filtered.nodes[node]['type'] == 'token' else G_filtered.degree(node) * scale_address + base_size)
        node_colors_filtered.append(color_map[G_filtered.nodes[node]['token_name']] if 'token_name' in G_filtered.nodes[node] else 'skyblue')
        edge_sizes_filtered.append((G_filtered.degree(node) -1) * 2 if G_filtered.nodes[node]['type'] != 'token' else 0)
    # Draw the filtered graph
    plt.figure(figsize=(15, 13))

    pos = nx.kamada_kawai_layout(G_filtered)      # ! Spiral layout looks fun
    nx.draw_networkx_nodes(G_filtered, pos, node_size=node_sizes_filtered, node_color=node_colors_filtered, alpha=0.7)
    # Modified part for drawing edges with token node colors
    for node1, node2 in G_filtered.edges():
        # Determine if node1 is a token node, else check if node2 is
        if G_filtered.nodes[node1]['type'] == 'token':
            edge_color = color_map[G_filtered.nodes[node1]['token_name']]
        elif G_filtered.nodes[node2]['type'] == 'token':
            edge_color = color_map[G_filtered.nodes[node2]['token_name']]
        else:
            edge_color = 'skyblue'  # Default color if no token nodes are involved
        
        # Draw this specific edge with the determined color
        nx.draw_networkx_edges(G_filtered, pos, edgelist=[(node1, node2)], width=1, alpha=0.5, edge_color=edge_color)

    # Since labels might make the graph cluttered, consider commenting out the label drawing line if not needed
    token_labels = {node: data['token_name'] for node, data in G_filtered.nodes(data=True) if data['type'] == 'token'}
    # Slightly adjust the label positions to reduce overlaps

    # Draw labels using the adjusted positions
    nx.draw_networkx_labels(G_filtered, pos, labels=token_labels, font_size=12, font_weight='bold')
    print(pos)
    # Legend for Token Names and Address Nodes, adapted for the filtered graph
    for token_name, color in color_map.items():
        plt.plot([0], [0], color=color, label=token_name, linestyle='None', marker='o')
    plt.plot([0], [0], color='skyblue', label='Address Nodes', linestyle='None', marker='o')
    #plt.legend(title="NFT Collections", loc='lower right')

    plt.title('Blockchain NFT Transaction Network')
    plt.axis('off')
    plt.show()

plot_network_nft()
# %%
