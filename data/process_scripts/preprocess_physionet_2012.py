from tqdm import tqdm
import os
import pandas as pd
import pickle


RAW_DATA_PATH = 'your_path/data/challenge-2012'

def read_ts(raw_data_path, set_name):
    ts = []
    pbar = tqdm(os.listdir(raw_data_path+'/set-'+ set_name), 
                desc='Reading time series set '+ set_name)
    for f in pbar:
        data = pd.read_csv(raw_data_path+'/set-'+set_name+'/'+f).iloc[1:]
        data = data.loc[data.Parameter.notna()]
        if len(data)<=5:
            continue
        data = data.loc[data.Value>=0] # neg Value indicates missingness.
        data['RecordID'] = int(f[:-4])
        ts.append(data)
    ts = pd.concat(ts)
    ts.Time = ts.Time.apply(lambda x:int(x[:2])*60
                            +int(x[3:])) # No. of minutes since admission.
    ts.rename(columns={'Time':'minute', 'Parameter':'variable', 
                       'Value':'value', 'RecordID':'ts_id'}, inplace=True)
    return ts


def read_outcomes(raw_data_path, set_name):
    oc = pd.read_csv(raw_data_path+'/Outcomes-'+set_name+'.txt', 
                     usecols=['RecordID', 'Length_of_stay', 'In-hospital_death'])
    oc['subset'] = set_name
    oc.RecordID = oc.RecordID.astype(int)
    oc.rename(columns={'RecordID':'ts_id', 'Length_of_stay':'length_of_stay', 
                       'In-hospital_death':'label'}, inplace=True)
    return oc


ts = pd.concat([read_ts(RAW_DATA_PATH, set_name) 
                for set_name in ['a','b','c']])
oc = pd.concat([read_outcomes(RAW_DATA_PATH, set_name) 
                for set_name in ['a','b','c']])

# Only keep outcomes for time series that exist.
ts_ids = ts.ts_id.unique()
oc = oc.loc[oc.ts_id.isin(ts_ids)]

# Drop duplicates.
ts = ts.drop_duplicates()

# Convert categorical to numeric.
ii = (ts.variable=='ICUType')
for val in [4,3,2,1]:
    kk = ii&(ts.value==val)
    ts.loc[kk, 'variable'] = 'ICUType_'+str(val)
ts.loc[ii, 'value'] = 1

# Drop subset column (no longer needed, splits are generated separately)
oc.drop(columns='subset', inplace=True)

# Sort by ts_id for consistency.
ts = ts.sort_values(['ts_id', 'minute']).reset_index(drop=True)
oc = oc.sort_values('ts_id').reset_index(drop=True)

# Store data (splits are now generated separately via generate_splits.py)
os.makedirs('../processed', exist_ok=True)
pickle.dump([ts, oc], 
            open('../processed/physionet_2012.pkl','wb'))
print(f'Saved {len(ts)} records, {len(oc)} outcomes')
print('Note: Run generate_splits.py to create train/val/test splits')