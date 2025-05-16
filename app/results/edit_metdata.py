from datetime import datetime
from pathlib import Path

import numpy as np


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


def ammend_column_to_metadata(metaf, column_value, index=-1):

    try:
        metadata = np.load(metaf, allow_pickle=True)
    except FileNotFoundError:
        raise RuntimeError

    if column_value is None:
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

        if dt < cutoff_date:  # before cutoff date
            row.append("app\\results\\microphone_states\\discrete_layout1.csv")
        elif dt < end_date:
            row.append("app\\results\\microphone_states\\gantry45.csv")
        else:
            pass
            # future ones will be added by the app

    metadata = np.array(metadata_list, dtype=object)

    # save
    np.save(metaf, metadata)


def fix_all():

    result_folder = Path("app/results/")
    # get all folders *.prop
    metafs = [
        f / "meta_data.npy"
        for f in result_folder.iterdir()
        if f.is_dir() and f.name.endswith(".prop")
    ]

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


def fix_microphone_angles(mic_states_folder):
    
    mic_states_folder = Path(mic_states_folder)

    for micf in mic_states_folder.glob("gantry*.csv"):

        micdata = np.loadtxt(micf, delimiter=",", skiprows=1, dtype=str)
        theta1 = float(micdata[2,2])
        theta2 = float(micdata[3,2])
        theta3 = float(micdata[4,2])
        dist = float(micdata[3,3])

        d1 = theta2 - theta1
        d2 = theta3 - theta2
        
        if np.isclose(dist, 1270):
            print("10bds skipping")
            continue
        elif np.isclose(d1, 1.0) and np.isclose(d2, 1.0):
            # fix the angles
            new_theta1 = theta2 - d1 * 1270 / dist
            new_theta3 = theta3 + d2 * 1270 / dist

            # update the angles in the file
            micdata[2, 2] = str(new_theta1)
            micdata[4, 2] = str(new_theta3)

        np.savetxt(micf, micdata, delimiter=",", fmt="%s")


if __name__ == "__main__":

    #print_metaf("app/results/5045_s30.prop/meta_data.npy")

    fix_microphone_angles('app/results/microphone_states/')

    # ammend_column_to_metadata('app/results/dalprop5045.prop/meta_data.npy', None)
