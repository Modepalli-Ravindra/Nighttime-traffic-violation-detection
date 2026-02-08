import os

def create_structure():
    dirs = [
        "weights",
        "input_videos",
        "output_videos",
        "src"
    ]
    
    for d in dirs:
        if not os.path.exists(d):
            os.makedirs(d)
            print(f"Created directory: {d}")
        else:
            print(f"Directory already exists: {d}")

if __name__ == "__main__":
    create_structure()
