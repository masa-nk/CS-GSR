# 各分析スクリプトをモジュールとしてインポート
import generate_data
import pca_analysis
import similarity_analysis

def run_analysis_pipeline():
    """
    GSR分析の全プロセスを順番に実行するメイン関数。
    """
    print("--- Step 1: Generating Virtual GSR Data ---")
    generate_data.main()
    print("\n" + "="*50 + "\n")

    print("--- Step 2: Performing PCA and Visualization ---")
    pca_analysis.perform_pca_and_visualize()
    print("\n" + "="*50 + "\n")

    print("--- Step 3: Performing Similarity Analysis with Random Forest ---")
    similarity_analysis.perform_similarity_analysis()
    print("\n" + "="*50 + "\n")

    print("--- GSR Analysis Pipeline Complete ---")
    # 出力されたファイルの場所をユーザーに通知
    print("Output files (method_A.csv, method_B.csv, pca_plot.png, confusion_matrix.png)")
    print("are saved in the 'gsr_analysis' directory.")

if __name__ == "__main__":
    run_analysis_pipeline()
