# 各分析スクリプトをモジュールとしてインポート
import pca_analysis
import similarity_analysis
import os

def run_user_data_analysis():
    """
    `user_data` ディレクトリ内のGSRデータを分析するメイン関数。
    """
    print("--- GSR Analysis Pipeline for User Data ---")

    # 分析対象のデータディレクトリを指定
    base_dir = os.path.dirname(os.path.abspath(__file__))
    user_data_directory = os.path.join(base_dir, 'user_data')

    if not os.path.isdir(user_data_directory):
        print(f"Error: Data directory not found at '{user_data_directory}'")
        print("Please create the 'user_data' directory and place your data files in it.")
        return

    print(f"Analyzing data from: {user_data_directory}")
    print("\n" + "="*50 + "\n")

    print("--- Step 1: Performing PCA and Visualization ---")
    pca_analysis.perform_pca_and_visualize(user_data_directory)
    print("\n" + "="*50 + "\n")

    print("--- Step 2: Performing Similarity Analysis with Random Forest ---")
    similarity_analysis.perform_similarity_analysis(user_data_directory)
    print("\n" + "="*50 + "\n")

    print("--- GSR Analysis Pipeline Complete ---")
    print("Output files (pca_plot.png, confusion_matrix.png)")
    print("are saved in the 'gsr_analysis' directory.")

if __name__ == "__main__":
    run_user_data_analysis()
