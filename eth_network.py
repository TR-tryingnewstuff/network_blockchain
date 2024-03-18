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

def get_accounts_name_df():

    files = glob.glob('accounts_eth/*.csv')

    accounts = []
    for f in files:

        accounts.append(pd.read_csv(f))

    df = pd.concat(accounts).dropna()

    return df 

wallets = get_accounts_name_df()


wallets['Name Tag'] = [re.sub('[0-9]', '', tag) for tag in wallets['Name Tag']] # ? Remove numbers from tag names
wallets = wallets.loc[np.invert(wallets['Name Tag'].str.contains('Fee Recipient'))] # ? Remove "Fee Recipient" Adress 
wallets['Name Tag'] = wallets['Name Tag'].apply(str.strip)  # ? Remove Leading / Trailing Whitespaces
wallets = wallets.set_index('Name Tag')
wallets

#%%

#%%

# Connect to Alchemy
API = 'CeaJwmD8u0qcSnGwuKPwbN_jGryIlD61'
alchemy = Alchemy(api_key=API)

accounts = wallets['Address']


all_transactions = {}

for account in accounts:
    transfers, page_key = alchemy.get_asset_transfers(from_address=account, max_count=100)
    all_transactions[account] = pd.DataFrame(transfers)

transactions = pd.concat(all_transactions)

#%%

df = transactions.reset_index().merge(wallets['Address'].reset_index(), left_on='level_0', right_on='Address')
df = df.drop(['level_0', 'level_1', 'erc721TokenId', 'erc1155Metadata', 'blockNum', 'uniqueId', 'hash', 'rawContract'], axis=1)
df


mapping_dict = df[['Name Tag', 'Address']].drop_duplicates().set_index('Address').to_dict()['Name Tag']

df['from'] =  df['from'].apply(lambda x : mapping_dict.get(x, None) if mapping_dict.get(x, None) != None else x)
df['to'] = df['to'].apply(lambda x : mapping_dict.get(x, None) if mapping_dict.get(x, None) != None else x)
df

df = df.loc[df['from'].apply(type) != type(None)]
df = df.loc[df['to'].apply(type) != type(None)]
#df = df.sample(frac=0.1)
print(len(df))

# %%

def is_informative(tag):
    
    return re.sub('0x', '', tag) == tag


G = nx.from_pandas_edgelist(df, 'from', 'to', edge_attr='value', edge_key='Name Tag')

informative_labels = {row['Name Tag']: row['Name Tag'] for index, row in df.iterrows() if is_informative(row['Name Tag'])}


# Node sizes and colors 
node_sizes = [G.degree(n) * 100 for n in G.nodes()]
node_colors = plt.cm.viridis(np.linspace(0, 1, len(G)))

# Edge widths based on the 'value' attribute, normalized for better visualization
values = nx.get_edge_attributes(G, 'value')
max_value = max(values.values()) if values else 1
edge_widths = [values.get(edge, 1) / max_value * 10 for edge in G.edges()]

# Layout for positioning
pos = nx.kamada_kawai_layout(G)

# Drawing the network
plt.figure(figsize=(12, 8))
nx.draw(G, pos, with_labels=True, labels=informative_labels, node_size=node_sizes, node_color=node_colors, edge_color='lightgray', width=edge_widths, font_size=8, font_weight='bold')

plt.title('Blockchain Transaction Network')
plt.axis('off')
plt.show()
#%%
