import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt
import os

def perform_similarity_analysis():
    """
    2つのGSRデータセットを読み込み、ランダムフォレストで分類し、
    その性能を評価することで類似性を分析する。
    """
    # --- データの読み込み ---
    current_dir = os.path.dirname(os.path.abspath(__file__))
    path_A = os.path.join(current_dir, 'method_A.csv')
    path_B = os.path.join(current_dir, 'method_B.csv')

    df_A = pd.read_csv(path_A)
    df_B = pd.read_csv(path_B)

    # 採取方法を識別する数値ラベルを追加 (A: 0, B: 1)
    df_A['Method'] = 0
    df_B['Method'] = 1

    # データを結合
    df_combined = pd.concat([df_A, df_B], ignore_index=True)

    # 特徴量とターゲットを分離
    X = df_combined.drop('Method', axis=1)
    y = df_combined['Method']

    # --- 訓練データとテストデータに分割 ---
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

    # --- ランダムフォレストモデルの訓練 ---
    rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_classifier.fit(X_train, y_train)

    # --- テストデータでの予測と評価 ---
    y_pred = rf_classifier.predict(X_test)

    # 評価指標の表示
    accuracy = accuracy_score(y_test, y_pred)
    print("--- Random Forest Classification Report ---")
    print(f"Accuracy: {accuracy:.4f}\n")

    print("Classification Report:")
    print(classification_report(y_test, y_pred, target_names=['Method A', 'Method B']))

    # 混同行列の可視化
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Predicted A', 'Predicted B'],
                yticklabels=['Actual A', 'Actual B'])
    plt.title('Confusion Matrix')
    plt.ylabel('Actual Label')
    plt.xlabel('Predicted Label')

    # 混同行列を画像ファイルとして保存
    output_path = os.path.join(current_dir, 'confusion_matrix.png')
    plt.savefig(output_path)
    print(f"\nConfusion matrix plot saved to {output_path}")

if __name__ == "__main__":
    # seabornも使うのでインストール
    try:
        import seaborn
    except ImportError:
        import pip
        pip.main(['install', 'seaborn'])
        import seaborn

    perform_similarity_analysis()
