import os
import shutil

def delete_folder(folder_path):
    if os.path.exists(folder_path):
        print(f"Deleting: {folder_path}")
        shutil.rmtree(folder_path)
    else:
        print(f"Not found: {folder_path}")

def clear_cache(root_dir="."):
    print("Clearing Streamlit and Python cache...\n")
    
    # Clear .streamlit cache folder
    streamlit_cache = os.path.join(os.path.expanduser("~"), ".streamlit", "cache")
    delete_folder(streamlit_cache)

    # Clear __pycache__ folders recursively
    for dirpath, dirnames, filenames in os.walk(root_dir):
        for dirname in dirnames:
            if dirname == "__pycache__":
                full_path = os.path.join(dirpath, dirname)
                delete_folder(full_path)

    print("\n✅ Cache clearing complete.")

if __name__ == "__main__":
    clear_cache()
