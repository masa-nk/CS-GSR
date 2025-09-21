import pandas as pd
import glob
import os

def load_user_data(data_dir):
    """
    指定されたディレクトリから、命名規則に従って複数のGSRデータCSVを読み込み、
    一つのDataFrameに統合する。

    Args:
        data_dir (str): CSVファイルが格納されているディレクトリのパス。

    Returns:
        pandas.DataFrame: 読み込まれたデータが統合されたDataFrame。
                         'Method'カラムに採取方法('A' or 'B')が格納される。
    """
    all_files_A = glob.glob(os.path.join(data_dir, 'method_A_*.csv'))
    all_files_B = glob.glob(os.path.join(data_dir, 'method_B_*.csv'))

    if not all_files_A and not all_files_B:
        raise FileNotFoundError(f"No data files matching 'method_A_*.csv' or 'method_B_*.csv' found in {data_dir}")

    li = []

    for f in all_files_A:
        df = pd.read_csv(f)
        df['Method'] = 'A'
        li.append(df)

    for f in all_files_B:
        df = pd.read_csv(f)
        df['Method'] = 'B'
        li.append(df)

    combined_df = pd.concat(li, axis=0, ignore_index=True)

    print(f"Loaded {len(all_files_A)} files for Method A and {len(all_files_B)} files for Method B.")

    return combined_df
