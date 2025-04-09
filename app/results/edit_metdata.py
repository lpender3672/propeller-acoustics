
import numpy as np

from pathlib import Path

from datetime import datetime

def clean_metadata(metaf):
    try:
        metadata = np.load(metaf, allow_pickle=True)
    except FileNotFoundError:
        raise FileNotFoundError

    # make all rows as long as the shortest row
    min_length = min(len(row) for row in metadata)
    metadata = [row[:min_length] for row in metadata]
    # save
    np.save(metaf, metadata)

def ammend_column_to_metadata(metaf, column_value, index = -1):

    try:
        metadata = np.load(metaf, allow_pickle=True)
    except FileNotFoundError:
        raise RuntimeError
    

    if column_value == None:        
        metadata = np.delete(metadata, index, axis=1)
        np.save(metaf, metadata)
        print(metadata)
        return

    else:

        if isinstance(column_value, (float, int)):
            new_column = column_value * np.ones((metadata.shape[0], 1))
        else:
            new_column = np.array([[column_value] for i in range(metadata.shape[0])])
        
        if index > -1 and index < metadata.shape[1]:
            metadata = np.insert(metadata, index, new_column, axis=1)

        else:
            metadata = np.hstack((metadata, new_column))
    
    np.save(metaf, metadata)
    print(metadata)
    

def add_correct_microphone(metaf):
    try:
        metadata = np.load(metaf, allow_pickle=True)
    except FileNotFoundError:
        raise RuntimeError
    
    if metadata.shape[1] >= 5:
        return
    
    cutoff_date = datetime(2025, 3, 25)
    end_date = datetime(2025, 4, 1)

    metadata_list = metadata.tolist()

    for row in metadata_list:
        audiof = Path(row[0])

        timestamp_str = audiof.stem.replace("audio_", "")
        dt = datetime.strptime(timestamp_str, "%Y-%m-%d-%H-%M-%S")
        
        if dt < cutoff_date: # before cutoff date
            row.append('app\\results\\microphone_states\\discrete_layout1.csv')
        elif dt < end_date:
            row.append('app\\results\\microphone_states\\gantry45.csv')
        else:
            pass
            # future ones will be added by the app

    metadata = np.array(metadata_list, dtype=object)
        
    # save
    np.save(metaf, metadata)


def fix_all():

    result_folder = Path('app/results/')
    # get all folders *.prop
    metafs = [f / 'meta_data.npy' for f in result_folder.iterdir() if f.is_dir() and f.name.endswith('.prop')]

    for mf in metafs:
        try:
            clean_metadata(mf)
        except FileNotFoundError:
            print(f"File not found: {mf}")
            continue

        add_correct_microphone(mf)


def print_metaf(metaf):
    try:
        metadata = np.load(metaf, allow_pickle=True)
    except FileNotFoundError:
        raise RuntimeError

    for row in metadata:
        print(row)

print_metaf('app/results/c.prop/meta_data.npy')


#ammend_column_to_metadata('app/results/dalprop5045.prop/meta_data.npy', None)

