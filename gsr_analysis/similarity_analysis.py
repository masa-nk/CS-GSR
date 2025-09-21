import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt
import os
from utils import load_user_data # ユーティリティ関数をインポート

def perform_similarity_analysis(data_dir, output_filename='confusion_matrix.png'):
    """
    指定されたディレクトリからGSRデータを読み込み、ランダムフォレストで分類し、
    その性能を評価することで類似性を分析する。
    """
    # --- データの読み込み ---
    try:
        df_combined = load_user_data(data_dir)
    except FileNotFoundError as e:
        print(e)
        return

    # 分類のためにMethodを数値に変換 (A: 0, B: 1)
    df_combined['Method'] = df_combined['Method'].map({'A': 0, 'B': 1})

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
    current_dir = os.path.dirname(os.path.abspath(__file__))
    output_path = os.path.join(current_dir, output_filename)
    plt.savefig(output_path)
    print(f"\nConfusion matrix plot saved to {output_path}")

if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.abspath(__file__))
    user_data_directory = os.path.join(base_dir, 'user_data')
    perform_similarity_analysis(user_data_directory)
