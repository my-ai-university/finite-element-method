import os
import shutil
import evaluate
import nltk


if __name__ == '__main__':
    # Remove the directory if it exists
    home_dir = os.path.expanduser("~")
    punkt_tab_path = os.path.join(home_dir, 'nltk_data', 'tokenizers', 'punkt_tab')
    if os.path.exists(punkt_tab_path):
        shutil.rmtree(punkt_tab_path)
        print(f"Removed existing directory: {punkt_tab_path}")

    # Retry loading the meteor metric
    evaluate.load("meteor")
    print("Successfully loaded the 'meteor' metric.")
    evaluate.load("rouge")
    print("Successfully loaded the 'rouge' metric.")
    evaluate.load("bertscore")
    print("Successfully loaded the 'bertscore' metric.")
