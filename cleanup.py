import os
import glob

# Paths to clean
INPUT_FOLDER = "/home/lubuharg/Documents/MTG/Input"
OUTPUT_FOLDER = "/home/lubuharg/Documents/MTG/Output"
DEBUG_FOLDER = "/home/lubuharg/Documents/MTG/debug_prepped"

def clean_folder(folder_path, extensions):
    for ext in extensions:
        for file in glob.glob(os.path.join(folder_path, f"*.{ext}")):
            try:
                os.remove(file)
                print(f"[CLEANED] {file}")
            except Exception as e:
                print(f"[ERROR] Could not delete {file}: {e}")

if __name__ == "__main__":
    print("[*] Cleaning up folders...")
    clean_folder(INPUT_FOLDER, ["jpg", "jpeg"])
    clean_folder(OUTPUT_FOLDER, ["png","csv"])
    clean_folder(DEBUG_FOLDER, ["png"])
    print("[✓] Cleanup complete.")
