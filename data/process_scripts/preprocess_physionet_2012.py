from tqdm import tqdm
import argparse
import os
import pandas as pd
import pickle


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROCESSED_PATH = os.path.join(SCRIPT_DIR, '..', 'processed')
parser = argparse.ArgumentParser(description='Preprocess PhysioNet Challenge 2012.')
parser.add_argument(
    '--raw_data_path',
    default=os.environ.get('P12_RAW_DATA', os.path.join(SCRIPT_DIR, '..', 'raw', 'challenge-2012')),
    help='Directory containing set-a/b/c and Outcomes-*.txt.',
)
RAW_DATA_PATH = parser.parse_args().raw_data_path

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
os.makedirs(PROCESSED_PATH, exist_ok=True)
output_path = os.path.join(PROCESSED_PATH, 'physionet_2012.pkl')
pickle.dump([ts, oc], open(output_path, 'wb'))
print(f'Saved {len(ts)} records, {len(oc)} outcomes')
print('Note: Run generate_splits.py to create train/val/test splits')