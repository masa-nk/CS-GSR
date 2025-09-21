import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os

def perform_pca_and_visualize():
    """
    GSRデータを読み込み、PCAを実行して3Dプロットを生成・保存する。
    """
    # --- データの読み込み ---
    current_dir = os.path.dirname(os.path.abspath(__file__))
    path_A = os.path.join(current_dir, 'method_A.csv')
    path_B = os.path.join(current_dir, 'method_B.csv')

    df_A = pd.read_csv(path_A)
    df_B = pd.read_csv(path_B)

    # 採取方法を識別するカラムを追加
    df_A['Method'] = 'A'
    df_B['Method'] = 'B'

    # データを結合
    df_combined = pd.concat([df_A, df_B], ignore_index=True)

    # 特徴量とターゲットを分離
    features = df_combined.drop('Method', axis=1)
    target = df_combined['Method']

    # --- データの前処理 ---
    # 標準化
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)

    # --- PCAの実行 ---
    pca = PCA(n_components=3)
    principal_components = pca.fit_transform(features_scaled)

    # PCAの結果をDataFrameに変換
    pca_df = pd.DataFrame(data=principal_components, columns=['PC1', 'PC2', 'PC3'])
    pca_df['Method'] = target

    # --- 可視化 ---
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    colors = {'A': 'blue', 'B': 'red'}

    for method, color in colors.items():
        subset = pca_df[pca_df['Method'] == method]
        ax.scatter(subset['PC1'], subset['PC2'], subset['PC3'], c=color, label=f'Method {method}', alpha=0.6)

    ax.set_title('PCA of GSR Data (3D Scatter Plot)')
    ax.set_xlabel('Principal Component 1')
    ax.set_ylabel('Principal Component 2')
    ax.set_zlabel('Principal Component 3')
    ax.legend()
    ax.grid(True)

    # プロットを画像ファイルとして保存
    output_path = os.path.join(current_dir, 'pca_plot.png')
    plt.savefig(output_path)
    print(f"PCA plot saved to {output_path}")

if __name__ == "__main__":
    perform_pca_and_visualize()
