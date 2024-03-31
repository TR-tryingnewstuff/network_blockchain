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
from sklearn.preprocessing import OrdinalEncoder
import networkx as nx
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
import numpy as np

cryptopunks_contract = '0xb47e3cd837dDF8e4c57F05d70Ab865de6e193BBB'

# Connect to Alchemy
API = 'CeaJwmD8u0qcSnGwuKPwbN_jGryIlD61'
alchemy = Alchemy(api_key=API)

contract_info = alchemy.get_contract_metadata(contract_address=cryptopunks_contract)


#%%

start_block = 3924495

def quick_process(l):
    d = l[0]
    
    return d ['tokenId']


all_transactions = {}
def get_owners_at_block(block):

    response = alchemy.get_owners_for_collection(contract_address=cryptopunks_contract, with_token_balances=True, block=block)

    if len(response['ownerAddresses']) > 0:
        df = pd.DataFrame(response['ownerAddresses'])
        df['tokenId'] = df['tokenBalances'].apply(quick_process)
        df = df.drop('tokenBalances', axis=1)

        df['block'] = np.repeat(block, len(df))

        return df

res = Parallel(4)(delayed(get_owners_at_block)(i) for i in range(start_block, start_block + 1000000, 10000))
#%%

files = glob.glob('/home/fast-pc-2023/Téléchargements/python/network_blockchain-main/crypto_punk_transaction/*')
d = []
for f in files:
    d.append(pd.read_csv(f).drop_duplicates(subset=['ownerAddress', 'tokenId']))

df = pd.concat(d)
df.to_parquet('reduced_crypto_punk_transaction.parquet', index=False)


#%%
df = pd.read_parquet('reduced_crypto_punk_transaction.parquet')
df = df.drop_duplicates(subset=['ownerAddress', 'tokenId'])

def get_nft_transaction_history(nft_id):


    curr_df = df.loc[df['tokenId'] == nft_id]
    #curr_df = curr_df.groupby('ownerAddress').first().reset_index()


    if len(curr_df) > 1:
        lagged = curr_df.shift(1)

        test = pd.concat([curr_df, lagged], axis=1)
        test.columns = ['to', 'tokenId', 'block_transac', 'from', 'token_id','block_prev']
        test.drop('token_id', axis=1)
        return test
        alls.append(test)

res = Parallel(10, batch_size=20)(delayed(get_nft_transaction_history)(nft_id) for nft_id in df['tokenId'].unique().tolist())
pd.concat(res)
#%%
to_network = pd.concat(res)
to_network.to_csv('final_crypto_punk_df.csv')
#%%

df = pd.read_parquet('reduced_crypto_punk_transaction.parquet')
