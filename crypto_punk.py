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
df = to_network.copy(deep=True)#.sample(frac=0.99)
df = df.T.drop_duplicates().T
n_nfts = df.groupby('from')['tokenId'].count()

nft_with_most_transac = (df['tokenId'].value_counts() > 5).replace(False, np.nan).dropna().index.to_list()
df = df.loc[df['tokenId'].isin(nft_with_most_transac)]


n_wallets_with_highest_n_transactions = df.groupby('to').count().sort_values('from').tail(1).index.to_list()
n_wallets_with_highest_n_transactions = (df['from'].value_counts() > 1).replace(False, np.nan).dropna().index.to_list()


df = df.loc[df['from'].isin(n_wallets_with_highest_n_transactions)]#.sample(frac=0.1)
df

#%%

df.sort_values(by=['block_prev'], inplace=True)

# Normalize both block_transac and block_prev to range [0, 1]
df['block_transac_normalized'] = (df['block_transac'] - df['block_transac'].min()) / (df['block_transac'].max() - df['block_transac'].min())
df['block_prev_normalized'] = (df['block_prev'] - df['block_prev'].min()) / (df['block_prev'].max() - df['block_prev'].min())

# Initialize the graph
G = nx.MultiDiGraph()

# Add edges from 'from' to 'tokenId' with block_transac_normalized as the time indicator
for _, row in df.iterrows():
    G.add_edge(row['to'], row['tokenId'], time=row['block_transac_normalized'])


# Prepare color map for nodes
color_map = []
node_size = []
for node in G:
    if len(node) == 42:  # Assuming wallets and token IDs can be differentiated by their length
        color_map.append('skyblue')  # Color for wallets
    else:
        color_map.append('lightgreen')  # Color for tokens
    # Node size based on degree, scaled for visualization
    node_size.append((G.degree(node) - 0.9) * 150)

# Prepare edge colors based on their 'time' attribute
edge_colors = [plt.cm.rainbow(G[u][v][0]['time']) for u, v in G.edges()]

# Use the Kamada-Kawai layout
pos = nx.spring_layout(G)

# Draw the graph
plt.figure(figsize=(14, 10))

nx.draw(G, pos, edge_color=edge_colors, node_color=color_map, node_size=node_size, with_labels=False, font_weight='bold', font_size=7, arrows=False)


plt.title("CryptoPunk NFT Transactions Network")
plt.colorbar(plt.cm.ScalarMappable(cmap=plt.cm.rainbow)).ax.set_title('Normalized Time', fontsize=15)


# Create a legend for the nodes
wallet_patch = mpatches.Patch(color='skyblue', label='Wallets')
nft_patch = mpatches.Patch(color='lightgreen', label='NFT')
plt.legend(handles=[wallet_patch, nft_patch], loc='upper left')
plt.show()
