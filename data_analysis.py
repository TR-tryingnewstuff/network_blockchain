#%%
import pandas as pd 
import matplotlib.pyplot as plt 
import numpy as np 
import plotly_express as px
import glob
import regex as re
# ! DATAFRAME ANALYSIS 
#%%
# ? ETH BLOCKCHAIN DATASET 
def get_wallet_info():
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
    return wallets

wallets = get_wallet_info()

df = pd.read_csv('eth_network_df.csv')
df = df.merge(wallets.reset_index(), right_on='Address', left_on='from')
# Calculate the frequency of each unique value in the 'asset' column and print it
print(f"\n Share of Top 10 assets exchanged :\n{(df['asset'].value_counts() / len(df)).head(10)}")
(df['asset'].value_counts() / len(df)).head(10).plot(kind='pie')
plt.show()

print(f"\n Top 10 Address share of transactions : \n{(df['Name Tag'].value_counts() / len(df)).head(10)}")
(df['Name Tag'].value_counts() / len(df)).head(30).plot(kind='pie')
plt.show()

# Calculate the frequency of each unique value in the 'category' column and print it
print(f"\n The 'category' column's value distribution as a proportion of the total dataset is:\n{df['category'].value_counts() / len(df)}")



# Calculate and print the number of unique sender addresses in the 'from' column
print(f"\n The number of unique sender addresses in the 'from' column is: {len(df['from'].unique())}")

# Calculate and print the number of unique recipient addresses in the 'to' column
print(f"\n The number of unique recipient addresses in the 'to' column is: {len(df['to'].unique())}")

# Calculate and print the number of unique blocks in the 'blockNum' column
print(f"\n The number of unique blocks in the 'blockNum' column is: {len(df['blockNum'].unique())}")


# Convert the 'blockNum' column from hexadecimal to decimal, then create and display a histogram of these values
block_min = df['blockNum'].apply(int, base=16).min()
block_max = df['blockNum'].apply(int, base=16).max()

print(block_max, block_min)
first_block = '2016-06-14'
last_block = '2024-03-21'

date_range = pd.date_range(start=first_block, end=last_block, periods=len(df['blockNum'].unique()))
blocks = df['blockNum'].sort_values().unique()
blocks.sort()
block_to_time_mapping = dict(map(lambda i,j : (i,j) , blocks,date_range))
df['time'] = df['blockNum'].apply(lambda x: block_to_time_mapping.get(x))
df['time'] = df['time'].dt.date

px.histogram(df.groupby(['time', 'Name Tag']).count().reset_index(), x='time', y='value', color='Name Tag')
#%%
df.groupby('time').count()['from'].rolling(30).mean().plot()
plt.title('Rolling Window of the number of transactions')
#%%
df.value_counts('asset').hist(bins=15)
plt.title('Histogram of the number of Transaction of a specific NFT')
#%%
df.value_counts('from').clip(0,1000000).hist(bins=100)
plt.title('Histogram of the number of NFT a Wallet received')

#%%


# ? ETH NFT DATASET 
df = pd.read_csv('nft_transactions_df.csv')
nft_collections = pd.read_csv('export-nft-top-contracts-1710592633736.csv')

df = df.merge(nft_collections, left_on='Unnamed: 0', right_on='Token')

# Calculate the frequency of each unique value in the 'asset' column and print it
print(f"\n Share of Top 10 assets exchanged :\n{(df['asset'].value_counts() / len(df)).head(10)}")
(df['asset'].value_counts() / len(df)).head(15).plot(kind='pie')
plt.show()


print(f"\n Top 10 Address share of transactions : \n{(df['Token Name'].value_counts() / len(df)).head(10)}")
(df['Token Name'].value_counts() / len(df)).head(15).plot(kind='pie')
plt.show()
# Calculate the frequency of each unique value in the 'category' column and print it
print(f"\n The 'category' column's value distribution as a proportion of the total dataset is:\n{df['category'].value_counts() / len(df)}")

# Calculate and print the number of unique sender addresses in the 'from' column
print(f"\n The number of unique sender addresses in the 'from' column is: {len(df['from'].unique())}")

# Calculate and print the number of unique recipient addresses in the 'to' column
print(f"\n The number of unique recipient addresses in the 'to' column is: {len(df['to'].unique())}")

# Calculate and print the number of unique blocks in the 'blockNum' column
print(f"\n The number of unique blocks in the 'blockNum' column is: {len(df['blockNum'].unique())}")

# Convert the 'blockNum' column from hexadecimal to decimal, then create and display a histogram of these values
block_min = df['blockNum'].apply(int, base=16).min()
block_max = df['blockNum'].apply(int, base=16).max()

print(f"The Minimum block is : {block_min} \n The Maximum block is : {block_max}")
first_block = '2020-11-27'
last_block = '2024-03-21'

date_range = pd.date_range(start=first_block, end=last_block, periods=len(df['blockNum'].unique()))
blocks = df['blockNum'].sort_values().unique()
blocks.sort()
block_to_time_mapping = dict(map(lambda i,j : (i,j) , blocks,date_range))
df['time'] = df['blockNum'].apply(lambda x: block_to_time_mapping.get(x))
df['time'] = df['time'].dt.date
print(df)

df = df.loc[df['Token Name'].isin((df['Token Name'].value_counts() > 100).replace(False, np.nan).dropna().index.to_list())]
px.histogram(df.groupby(['Token Name','time'])['from'].count().reset_index().sort_values('time'),log_y=True, x='time', y='from', color='Token Name', nbins=1000)
#%%
#df.groupby('time').count()['from'].rolling(30).mean().plot(rot=30)
df.value_counts(['Token Name', 'tokenId']).hist(bins=7)
plt.title('Histogram of the number of Transaction of a specific NFT')
#%%

df.value_counts('to').clip(0,40).hist(bins=40, log=True)
plt.title('Histogram of the number of NFT a Wallet received')


#%%

# ? Crypto Punk NFT DATASET 
df = pd.read_csv('final_crypto_punk_df.csv')
df

# Calculate and print the number of unique sender addresses in the 'from' column
print(f"\n The number of unique sender addresses in the 'from' column is: {len(df['from'].unique())}")

# Calculate and print the number of unique recipient addresses in the 'to' column
print(f"\n The number of unique recipient addresses in the 'to' column is: {len(df['to'].unique())}")

# Calculate and print the number of unique blocks in the 'blockNum' column
print(f"\n The number of unique blocks in the 'blockNum' column is: {len(df['block_transac'].unique())}")

# Calculate and print the number of unique blocks in the 'blockNum' column
print(f"\n The number of unique NFT is: {len(df['tokenId'].unique())}")

# Convert the 'blockNum' column from hexadecimal to decimal, then create and display a histogram of these values
block_min = df['block_transac'].apply(int).min()
block_max = df['block_transac'].apply(int).max()


print(f"\nThe Minimum block is : {block_min} \nThe Maximum block is : {block_max}")
first_block = '2017-05-24'
last_block = '2024-03-24'

date_range = pd.date_range(start=first_block, end=last_block, periods=len(df['block_transac'].unique()))
blocks = df['block_transac'].sort_values().unique()
blocks.sort()
block_to_time_mapping = dict(map(lambda i,j : (i,j) , blocks,date_range))
df['time'] = df['block_transac'].apply(lambda x: block_to_time_mapping.get(x))
df['time'] = df['time'].dt.date


(df['time'].value_counts(sort='descending') / len(df)).head(20).plot(kind='pie')
plt.show()

#df.groupby('time').count()['from'].rolling(30).mean().plot()
df.value_counts('tokenId').hist(bins=15)
plt.title('Histogram of the number of Transaction of a specific NFT')
#%%
df.value_counts('to').clip(0,40).hist(bins=40, log=True)
plt.title('Histogram of the number of NFT a Wallet received')

# %%
df