import pandas as pd
import numpy as np
import os

def generate_gsr_data(num_samples, means, std_devs, file_path, elements):
    """
    指定された平均と標準偏差に基づいてGSRデータを生成し、CSVファイルに保存する。
    """
    # 10元素のデータフレームを生成
    data = {}
    for i, element in enumerate(elements):
        # 平均と標準偏差が0でない場合のみ、正規分布に従う乱数を生成
        if means[i] > 0 and std_devs[i] > 0:
            data[element] = np.random.normal(loc=means[i], scale=std_devs[i], size=num_samples)
        else:
            # 平均または標準偏差が0の場合は、その値を定数として設定
            data[element] = np.full(num_samples, means[i])

    df = pd.DataFrame(data)

    # 成分比なので、負の値を0にクリップし、合計が100になるように正規化
    df[df < 0] = 0
    df = df.div(df.sum(axis=1), axis=0) * 100

    # ファイルに保存
    df.to_csv(file_path, index=False)
    print(f"Generated data and saved to {file_path}")

def main():
    """
    仮想GSRデータを生成するメイン関数。
    """
    # 仮想データの出力先ディレクトリ
    output_dir = os.path.dirname(os.path.abspath(__file__))

    # 10種類の元素
    elements = ['Pb', 'Ba', 'Sb', 'Zn', 'Ti', 'Ca', 'Si', 'Sn', 'Cu', 'Al']

    # --- 採取方法Aのパラメータ ---
    # Pb, Ba, Sbを特徴的に高く設定
    means_A =     [35, 30, 15, 2, 1, 5, 5, 2, 3, 2]
    std_devs_A =  [5,  5,  5,  1, 0.5, 2, 2, 1, 1, 1]

    # --- 採取方法Bのパラメータ ---
    # Pb, Ba, Sbの平均がやや低く、ばらつきが大きい設定
    means_B =     [30, 25, 12, 2, 1, 6, 6, 2, 3, 3]
    std_devs_B =  [8,  8,  8,  1, 0.5, 2.5, 2.5, 1, 1, 1]

    # データ生成
    generate_gsr_data(
        num_samples=500,
        means=means_A,
        std_devs=std_devs_A,
        file_path=os.path.join(output_dir, 'method_A.csv'),
        elements=elements
    )

    generate_gsr_data(
        num_samples=500,
        means=means_B,
        std_devs=std_devs_B,
        file_path=os.path.join(output_dir, 'method_B.csv'),
        elements=elements
    )

if __name__ == "__main__":
    main()
