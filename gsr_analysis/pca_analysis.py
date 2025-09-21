import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os
from utils import load_user_data # ユーティリティ関数をインポート

def perform_pca_and_visualize(data_dir, output_filename='pca_plot.png'):
    """
    指定されたディレクトリからGSRデータを読み込み、PCAを実行して3Dプロットを生成・保存する。
    """
    # --- データの読み込み ---
    try:
        df_combined = load_user_data(data_dir)
    except FileNotFoundError as e:
        print(e)
        return

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
    current_dir = os.path.dirname(os.path.abspath(__file__))
    output_path = os.path.join(current_dir, output_filename)
    plt.savefig(output_path)
    print(f"PCA plot saved to {output_path}")

if __name__ == "__main__":
    # このスクリプトを直接実行する場合のデフォルトの動作
    base_dir = os.path.dirname(os.path.abspath(__file__))
    user_data_directory = os.path.join(base_dir, 'user_data')
    perform_pca_and_visualize(user_data_directory)
