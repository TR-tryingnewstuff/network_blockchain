#%%
import pandas as pd
import numpy as np 
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.patches as mpatches

#%%
to_network = pd.read_csv('final_crypto_punk_df.csv')
to_network = to_network.sort_values('block_transac')
#%%
df = to_network.copy(deep=True)
df = df.T.drop_duplicates().T
n_nfts = df.groupby('from')['tokenId'].count()

nft_with_most_transac = (df['tokenId'].value_counts() > 4).replace(False, np.nan).dropna().index.to_list()


#df = df.loc[df['tokenId'].isin(nft_with_most_transac)]
from_to_keep_token = (df.groupby(['from']).count() > 6)['to'].replace(False, np.nan).dropna().index.to_list()
token_to_keep = df.loc[df['from'].isin(from_to_keep_token)]['tokenId'].to_list()
df = df.loc[df['tokenId'].isin(token_to_keep)]
df = df.loc[df['tokenId'].isin(nft_with_most_transac)]


n_wallets_with_highest_n_transactions = (df['from'].value_counts() > 1).replace(False, np.nan).dropna().index.tolist()
#df = df.loc[df['from'].isin(n_wallets_with_highest_n_transactions)]
df['tokenId'] = df['tokenId'].apply(int, base=16)

df

df.sort_values(by=['block_prev'], inplace=True)

# Normalize both block_transac and block_prev to range [0, 1]
df['block_transac_normalized'] = (df['block_transac'] - df['block_transac'].min()) / (df['block_transac'].max() - df['block_transac'].min())
df['block_prev_normalized'] = (df['block_prev'] - df['block_prev'].min()) / (df['block_prev'].max() - df['block_prev'].min())

# Initialize the graph
G = nx.MultiDiGraph()

# Add edges from 'from' to 'tokenId' with block_transac_normalized as the time indicator
for _, row in df.iterrows():
    G.add_edge(row['from'], row['to'], token=row['tokenId'] / df['tokenId'].max())


# Prepare color map for nodes
color_map = []
node_size = []

for node in G:
  # Assuming wallets and token IDs can be differentiated by their length


    if (G.in_degree(node) > G.out_degree(node)):
        color_map.append('cyan')
        node_size.append(((G.degree(node)) * 20 + 20))

    elif G.out_degree(node) > 3:
        color_map.append('black')

        node_size.append(((G.degree(node)) * 20 + 20))        
    
    else:
        col = list(plt.cm.hot(df.loc[df['to'] == node]['tokenId'].min() / df['tokenId'].max()))
        col[-1] = 0.4

              # Color for wallets
        color_map.append(col)
        node_size.append(((G.degree(node)- 1) * 20 + 10))


    # Node size based on degree, scaled for visualization
    

# Prepare edge colors based on their 'time' attribute
edge_colors = [plt.cm.hot(G[u][v][0]['token']) for u, v in G.edges()]

pos = nx.random_layout(G)

for k in pos.keys():
    try:
        pos[k][0] = (df.loc[df['from'] == k].groupby('from').first()['block_prev_normalized'].clip(0.3, 1) - 0.25)
        pos[k][1] = df.loc[df['from'] == k].groupby('from').first()['tokenId'] / df['tokenId'].max()
    except:

        pos[k] = np.array([0, 0.5 ])

# Draw the graph

plt.figure(figsize=(10, 10))

nx.draw(G, pos, edge_color=edge_colors,node_color=color_map, width=0.4, node_size=node_size, with_labels=False, arrows=True)

first_block = '2017-05-24'
last_block = '2024-03-24'

date_range = pd.date_range(start=first_block, end=last_block, periods=100).to_list()

plt.title("CryptoPunk NFT Transactions Network")
wallet_patch = mpatches.Patch(color='cyan', label='Collectors')
nft_patch = mpatches.Patch(color='black', label='Resalers')
plt.legend(handles=[wallet_patch, nft_patch], loc='upper left')


plt.draw()
