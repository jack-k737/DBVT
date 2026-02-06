from tqdm import tqdm
import pandas as pd
import numpy as np
import pickle
import os


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RAW_DATA_PATH = 'your_path/data/challenge-2019'
PROCESSED_PATH = os.path.join(SCRIPT_DIR, '..', 'processed')
SUBSETS = ['training_setA', 'training_setB', 'training_setC']
TIME_COL = 'ICULOS'
LABEL_COL = 'SepsisLabel'
STATIC_VARS = ['Age', 'Gender', 'Unit1', 'Unit2', 'HospAdmTime']


def safe_float(val):
    """Safely cast values to float, returning NaN if conversion fails."""
    try:
        if pd.isna(val):
            return -1.0
        return float(val)
    except Exception:
        return np.nan


def collect_patient_files():
    files = []
    for subset in SUBSETS:
        subset_path = os.path.join(RAW_DATA_PATH, subset)
        if not os.path.isdir(subset_path):
            continue
        for fname in os.listdir(subset_path):
            if fname.endswith('.psv'):
                files.append((subset, os.path.join(subset_path, fname)))
    files.sort(key=lambda x: x[1])
    return files


def add_static_event(events, ts_id, var_name, raw_value):
    val = safe_float(raw_value)
    if np.isnan(val):
        val = -1.0
    events.append((ts_id, 0, var_name, float(val)))


patient_files = collect_patient_files()
if len(patient_files) == 0:
    raise FileNotFoundError(f'No Physionet 2019 files found under {RAW_DATA_PATH}')

example_df = pd.read_csv(patient_files[0][1], sep='|', nrows=1)
all_cols = list(example_df.columns)
required_cols = STATIC_VARS + [TIME_COL, LABEL_COL]
missing = [col for col in required_cols if col not in all_cols]
if missing:
    raise ValueError('Missing expected columns: ' + str(missing))
ts_params = [col for col in all_cols if col not in required_cols]
print(f'# time series variables: {len(ts_params)}')

records = []
oc_records = []
skipped = 0

for subset, file_path in tqdm(patient_files, desc='Processing Physionet2019'):
    df = pd.read_csv(file_path, sep='|')
    if LABEL_COL not in df:
        df[LABEL_COL] = 0
    df[LABEL_COL] = df[LABEL_COL].fillna(0)

    ts_id = os.path.splitext(os.path.basename(file_path))[0]
    if ts_id.startswith('p'):
        ts_id = ts_id[1:]
    ts_id = int(ts_id)

    ts_values = df[[TIME_COL] + ts_params].copy()
    ts_values[TIME_COL] = ts_values[TIME_COL].apply(safe_float)
    ts_values = ts_values.dropna(subset=[TIME_COL])
    if ts_values.empty:
        skipped += 1
        continue
    start_hour = ts_values[TIME_COL].min()
    ts_values[TIME_COL] = ts_values[TIME_COL] - start_hour

    ts_long = ts_values.set_index(TIME_COL)[ts_params].stack().reset_index()
    ts_long.columns = [TIME_COL, 'variable', 'value']
    ts_long['value'] = ts_long['value'].apply(safe_float)
    ts_long = ts_long.loc[ts_long['value'].notna()]
    ts_long[TIME_COL] = ts_long[TIME_COL].apply(safe_float)
    ts_long = ts_long.loc[ts_long[TIME_COL].notna()]
    ts_long['minute'] = (ts_long[TIME_COL] * 60).round().astype(int)
    ts_long = ts_long.loc[ts_long['minute'] >= 0]
    if ts_long.empty:
        skipped += 1
        continue

    # Determine sepsis label and cutoff time based on SepsisLabel
    # SepsisLabel=1 means t >= t_sepsis - 6, i.e., sepsis will occur within 6 hours
    label_df = df[[TIME_COL, LABEL_COL]].copy()
    label_df[TIME_COL] = label_df[TIME_COL].apply(safe_float)
    label_df = label_df.dropna(subset=[TIME_COL])
    label_df[TIME_COL] = label_df[TIME_COL] - start_hour
    label_df[LABEL_COL] = label_df[LABEL_COL].fillna(0)
    
    # Check if patient develops sepsis (has any SepsisLabel=1)
    sepsis_any = int(label_df[LABEL_COL].max())
    
    if sepsis_any == 1:
        first_sepsis_hour = label_df.loc[label_df[LABEL_COL] == 1, TIME_COL].iloc[0]
        cutoff_minute = int(round(first_sepsis_hour * 60))
        if cutoff_minute < 0:
            cutoff_minute = 0
        # Truncate time series data: keep only data BEFORE the first SepsisLabel=1
        # This prevents data leakage - we don't use any data from the sepsis warning period
        ts_long = ts_long[ts_long['minute'] < cutoff_minute]
        if ts_long.empty:
            skipped += 1
            continue
        onset_minute = cutoff_minute
    else:
        # Non-sepsis patient: use all data, label is 0
        onset_minute = -1
    
    ts_long['ts_id'] = ts_id

    static_events = []
    first_row = df.iloc[0]
    add_static_event(static_events, ts_id, 'Age', first_row.get('Age'))
    add_static_event(static_events, ts_id, 'Gender', first_row.get('Gender'))
    add_static_event(static_events, ts_id, 'Unit1', first_row.get('Unit1'))
    add_static_event(static_events, ts_id, 'Unit2', first_row.get('Unit2'))
    add_static_event(static_events, ts_id, 'HospAdmTime', -first_row.get('HospAdmTime'))

    records.extend(static_events)
    records.extend(list(ts_long[['ts_id', 'minute', 'variable', 'value']].itertuples(index=False, name=None)))

    # Label: represents "sepsis after 6 hours"
    oc_records.append({'ts_id': ts_id,
                       'label': sepsis_any})

print(f'Skipped patients without usable data: {skipped}')

data = pd.DataFrame(records, columns=['ts_id', 'minute', 'variable', 'value'])
data.drop_duplicates(inplace=True)
data['ts_id'] = data['ts_id'].astype(int)
data['minute'] = data['minute'].astype(int)
data['value'] = data['value'].astype(float)

oc = pd.DataFrame(oc_records)
oc = oc.loc[oc.ts_id.isin(data.ts_id.unique())].reset_index(drop=True)

# Sort by ts_id for consistency.
data = data.sort_values(['ts_id', 'minute']).reset_index(drop=True)
oc = oc.sort_values('ts_id').reset_index(drop=True)

# Store data (splits are now generated separately via generate_splits.py)
os.makedirs(PROCESSED_PATH, exist_ok=True)
output_path = os.path.join(PROCESSED_PATH, 'physionet_2019.pkl')
pickle.dump([data, oc], open(output_path, 'wb'))
print(f'Saved {len(data)} records, {len(oc)} outcomes to {output_path}')
print('Note: Run generate_splits.py to create train/val/test splits')