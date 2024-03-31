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


df['tokenId'] = df['tokenId'].apply(int, base=16)

df

df.sort_values(by=['block_prev'], inplace=True)

# Normalize both block_transac and block_prev to range [0, 1]
df['block_transac_normalized'] = (df['block_transac'] - df['block_transac'].min()) / (df['block_transac'].max() - df['block_transac'].min())
df['block_prev_normalized'] = (df['block_prev'] - df['block_prev'].min()) / (df['block_prev'].max() - df['block_prev'].min())



# %%
# Add edges from 'from' to 'tokenId' with block_transac_normalized as the time indicator

G = nx.Graph(G)
for _, row in df.iterrows():
    G.add_edge(row['from'], row['to'], token=row['tokenId'] / df['tokenId'].max())

# Get number of nodes
num_nodes = G.number_of_nodes()

# Get number of edges
num_edges = G.number_of_edges()

# Get density
density = nx.density(G)

# Get average degree
avg_degree = np.mean([val for (node, val) in G.degree()])

# Get clustering coefficient
clustering_coefficient = nx.average_clustering(G)

# Get assortativity
assortativity = nx.degree_assortativity_coefficient(G)

# Get median degree
degree_values = [val for (node, val) in G.degree()]
median_degree = np.median(degree_values)

# Get percentage of connected components
percentage_connected_components = nx.number_connected_components(G) / num_nodes


# Get edge weights
edge_weights = [weight['token'] for _, _, weight in G.edges(data=True)]

# Calculate edge statistics
edge_stats = {
    'Minimum': np.min(edge_weights),
    'Maximum': np.max(edge_weights),
    'Median': np.median(edge_weights),
    'Mean': np.mean(edge_weights),
    'Standard Deviation': np.std(edge_weights)
}

# Print the obtained information
print("Number of Nodes:", num_nodes)
print("Number of Edges:", num_edges)
print("Density:", density)
print("Average Degree:", avg_degree)
print("Clustering Coefficient:", clustering_coefficient)
print("Assortativity:", assortativity)
print("Median Degree:", median_degree)
print("Percentage of Connected Components:", percentage_connected_components)

print("Blockchain transactions Edges Statistics:")
for stat, value in edge_stats.items():
    print(f"{stat}: {value}")# %%

# %%
# Compute closeness centrality
closeness_centralities = nx.closeness_centrality(G)

# Compute farness centrality
farness_centralities = {node: 1 / value for node, value in closeness_centralities.items()}

# Convert centralities to lists
closeness_values = list(closeness_centralities.values())
farness_values = list(farness_centralities.values())

# Compute summary statistics for closeness centrality
closeness_mean = np.mean(closeness_values)
closeness_std = np.std(closeness_values)

# Compute summary statistics for farness centrality
farness_mean = np.mean(farness_values)
farness_std = np.std(farness_values)

# Print summary statistics for closeness centrality
print("Closeness Centrality Summary:")
print(f"Mean: {closeness_mean}")
print(f"Standard Deviation: {closeness_std}")

# Print summary statistics for farness centrality
print("\nFarness Centrality Summary:")
print(f"Mean: {farness_mean}")
print(f"Standard Deviation: {farness_std}")

# %%


# Number of nodes
num_nodes = len(G.nodes())

# Number of edges
num_edges = len(G.edges())

# Get degrees of all nodes
degrees = list(dict(G.degree()).values())

# Calculate maximum, minimum, median, and standard deviation of degrees
max_degree = np.max(degrees)
min_degree = np.min(degrees)
median_degree = np.median(degrees)
std_dev_degree = np.std(degrees)

# Print the statistics
print("Statistics about Nodes:")
print("- Number of Nodes:", num_nodes)

print("\nStatistics about Edges:")
print("- Number of Edges:", num_edges)

print("\nDegree Statistics:")
print("- Maximum Degree:", max_degree)
print("- Minimum Degree:", min_degree)
print("- Median Degree:", median_degree)
print("- Standard Deviation of Degree:", std_dev_degree)
# %%
# Compute degree centrality
degree_centralities = nx.degree_centrality(G)

# Sort nodes by degree centrality in descending order
sorted_degrees = sorted(degree_centralities.items(), key=lambda x: x[1], reverse=True)

# Print top 10 most connected nodes
print("Top 10 Most Connected Nodes:")
for node, degree_centrality in sorted_degrees[:10]:
    print(f"Node: {node}, Degree Centrality: {degree_centrality}")

# Plot degree distribution
plt.figure(figsize=(10, 6))
plt.hist(list(degree_centralities.values()), bins=20, color='skyblue', edgecolor='black')
plt.title('Degree Centrality Distribution')
plt.xlabel('Degree Centrality')
plt.ylabel('Frequency')
plt.grid(True)
plt.show()

#%%
# Define the number of top connected nodes you want to visualize
num_top_nodes = 5

# Create subplots
fig, axes = plt.subplots(num_top_nodes, 1, figsize=(10, 8*num_top_nodes))

# Iterate over the top connected nodes
for i, (node, _) in enumerate(sorted_degrees[:num_top_nodes]):
    # Get the neighbors of the current node
    neighbors = list(G.neighbors(node))
    
    # Draw the subgraph with the current node and its neighbors
    subgraph = G.subgraph([node] + neighbors)
    
    # Draw the subgraph
    pos = nx.spring_layout(subgraph)
    nx.draw(subgraph, pos, ax=axes[i], with_labels=False, node_color='skyblue', node_size=3000, edge_color='gray', width=2)
    axes[i].set_title(f'Subgraph with Node {node} and Its Neighbors')
    axes[i].set_axis_off()

# Adjust layout
plt.tight_layout()
plt.show()

# %%
num_top_nodes = 5

# Create a dictionary to store neighbors of top connected nodes
neighbor_counts = {}

# Iterate over the top connected nodes
for node, _ in sorted_degrees[:num_top_nodes]:
    # Get the neighbors of the current node
    neighbors = list(G.neighbors(node))
    
    # Update neighbor counts
    for neighbor in neighbors:
        neighbor_counts[neighbor] = neighbor_counts.get(neighbor, 0) + 1

sorted_neighbors = sorted(neighbor_counts.items(), key=lambda x: x[1], reverse=True)

# Print analysis
print("\nAnalysis of Top Connected Nodes and Their Neighbors:")

for i, (node, _) in enumerate(sorted_degrees[:num_top_nodes]):
    print(f"\nTop Connected Node {i+1}: {node}")
    print("Neighbors:")
    neighbors = list(G.neighbors(node))
    for neighbor in neighbors:
        print(f"- {neighbor}")
    
    # Check if any neighbors are also top connected nodes
    top_neighbors = [n for n in neighbors if n in [n[0] for n in sorted_degrees]]
    if len(top_neighbors) > 0:
        print("Top Connected Neighbors:")
        for top_neighbor in top_neighbors:
            print(f"- {top_neighbor}")
    else:
        print("No Top Connected Neighbors")

# Print top recurring neighbors and their counts
print("\nTop Recurring Neighbors:")
for neighbor, count in sorted_neighbors[:20]:
    print(f"Node: {neighbor}, Occurrences: {count}")

# %%
# Identify common neighbors among the top connected nodes
common_neighbors = {}
for i, (node, _) in enumerate(sorted_degrees[:num_top_nodes]):
    neighbors = set(G.neighbors(node))
    for neighbor in neighbors:
        common_neighbors[neighbor] = common_neighbors.get(neighbor, 0) + 1

# Sort common neighbors by their occurrence counts
sorted_common_neighbors = sorted(common_neighbors.items(), key=lambda x: x[1], reverse=True)

# Print common neighbors and their counts
print("\nCommon Neighbors among Top Connected Nodes:")
for neighbor, count in sorted_common_neighbors[:10]:
    print(f"Node: {neighbor}, Common Neighbors Count: {count}")

# Analyze attributes of common neighbors
if 'attribute_column' in G.nodes:
    attribute_counts = {}
    for neighbor, _ in sorted_common_neighbors:
        if 'attribute_column' in G.nodes[neighbor]:
            attribute = G.nodes[neighbor]['attribute_column']
            attribute_counts[attribute] = attribute_counts.get(attribute, 0) + 1

    # Print attribute counts
    print("\nAttribute Analysis of Common Neighbors:")
    for attribute, count in attribute_counts.items():
        print(f"Attribute: {attribute}, Count: {count}")
else:
    print("\nNo attribute column found in the network.")

# %%
# Group transactions by tokenId (CryptoPunk)
punk_transactions = df.groupby('tokenId')

# Iterate over each CryptoPunk's transactions
for punk_id, transactions in punk_transactions:
    print(f"\nCryptoPunk ID: {punk_id}")
    
    # Analyze transaction patterns for this CryptoPunk
    num_transactions = len(transactions)
    if num_transactions < 2:
        print("Insufficient transactions for analysis.")
        continue
    
    # Sort transactions by block_transac for sequential analysis
    transactions = transactions.sort_values(by='block_transac')
    
    # Calculate time between consecutive transactions
    time_gaps = transactions['block_transac'].diff().dropna()
    
    print(f"Number of Transactions: {num_transactions}")
    print("Time between Consecutive Transactions (in blocks):")
    print(time_gaps)
    
    # Perform statistical analysis on time gaps, or any other relevant analysis
    
    # Example: Calculate mean and standard deviation of time gaps
    mean_time_gap = time_gaps.mean()
    std_dev_time_gap = time_gaps.std()
    
    print(f"Mean Time Gap: {mean_time_gap}")
    print(f"Standard Deviation of Time Gap: {std_dev_time_gap}")

# %%
# Group transactions by tokenId (CryptoPunk)
punk_transactions = df.groupby('tokenId')

# Iterate over each CryptoPunk's transactions
for punk_id, transactions in punk_transactions:
    print(f"\nCryptoPunk ID: {punk_id}")
    
    # Analyze transaction patterns for this CryptoPunk
    num_transactions = len(transactions)
    if num_transactions < 2:
        print("Insufficient transactions for analysis.")
        continue
    
    # Sort transactions by block_transac for sequential analysis
    transactions = transactions.sort_values(by='block_transac')
    
    # Calculate time between consecutive transactions
    time_gaps = transactions['block_transac'].diff().dropna()
    
    print(f"Number of Transactions: {num_transactions}")
    
    if len(time_gaps) > 0:
        # Calculate mean and standard deviation of time gaps
        mean_time_gap = time_gaps.mean()
        std_dev_time_gap = time_gaps.std()
        
        # Print summary
        print(f"Mean Time Gap between Transactions: {mean_time_gap:.2f} blocks")
        print(f"Standard Deviation of Time Gap: {std_dev_time_gap:.2f} blocks")
    else:
        print("No consecutive transactions for analysis.")

# %%
import pandas as pd

# Initialize an empty list to store summary information
summary_data = []

# Group transactions by tokenId (CryptoPunk)
punk_transactions = df.set('from')

# Iterate over each CryptoPunk's transactions
for punk_id, transactions in punk_transactions:
    # Initialize summary dictionary for the current CryptoPunk
    summary = {'CryptoPunk ID': punk_id}
    
    # Analyze transaction patterns for this CryptoPunk
    num_transactions = len(transactions)
    summary['Number of Transactions'] = num_transactions
    

    # Sort transactions by block_transac for sequential analysis
    transactions = transactions.sort_values(by='block_transac')
    
    # Calculate time between consecutive transactions
    time_gaps = transactions['block_transac'].diff().dropna()
    
    # Calculate mean and standard deviation of time gaps
    mean_time_gap = time_gaps.mean()
    std_dev_time_gap = time_gaps.std()
    
    summary['Mean Time Gap between Transactions'] = mean_time_gap
    summary['Standard Deviation of Time Gap'] = std_dev_time_gap
    
    # Append summary to the list
    summary_data.append(summary)

# Create a DataFrame from the summary data
summary_df = pd.DataFrame(summary_data)

# Print the summary DataFrame
print(summary_df)

# Generate overall summary statistics
overall_summary = {
    'Mean Number of Transactions': summary_df['Number of Transactions'].mean(),
    'Mean Mean Time Gap': summary_df['Mean Time Gap between Transactions'].mean(),
    'Mean Standard Deviation of Time Gap': summary_df['Standard Deviation of Time Gap'].mean()
}

# Print overall summary statistics
print("\nOverall Summary:")
for key, value in overall_summary.items():
    print(f"{key}: {value:.2f}")


#%%

first_block = '2017-05-24'
last_block = '2024-03-24'

date_range = pd.date_range(start=first_block, end=last_block, periods=len(df['block_transac'].unique()))
blocks = df['block_transac'].sort_values().unique()
blocks.sort()
block_to_time_mapping = dict(map(lambda i,j : (i,j) , blocks,date_range))
df['time'] = df['block_transac'].apply(lambda x: block_to_time_mapping.get(x))
df['time'] = df['time'].dt.date
df

#%%

df['from'].nunique()
#%%
pattern = df.sort_values('block_transac').groupby('from')['time'].apply(pd.Series.diff).dropna().apply(lambda x: int(str(x).split()[0]))

large_exchange = (pattern.reset_index()['from'].value_counts() > 4).replace(False, np.nan).dropna().index.to_list()
pattern = pattern.reset_index().loc[pattern.reset_index()['from'].isin(large_exchange)]
dict_n_transac = pattern.groupby('from').count()['time'].to_dict()
pattern['n_transac'] =  pattern['from'].apply(lambda x: dict_n_transac.get(x))

from sklearn.preprocessing import OrdinalEncoder
pattern.sort_values('n_transac', inplace=True)
pattern['from'] = pattern['from'].apply(lambda x : x[0:6])
import plotly_express as px
px.box(pattern.sort_values('n_transac'), x='from',y='time', color='n_transac', width=1000)

#%%


#%%
summary_df['standardized_time'] = summary_df['Mean Time Gap between Transactions'] * 15 / (60 * 60 * 24)

px.histogram(df.dropna().reset_index(), x='standardized_time', color='Number of Transactions')
#%%

summary_df.dropna().reset_index()['standardized_time']