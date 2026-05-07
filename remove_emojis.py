import os
import emoji

def remove_emojis_from_file(filepath):
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
    except Exception as e:
        print(f"Skipping {filepath} due to read error: {e}")
        return

    # Check if there are emojis to avoid unnecessary writes
    if emoji.emoji_count(content) > 0:
        cleaned_content = emoji.replace_emoji(content, replace='')
        
        # We also want to remove some decorative unicode symbols if emoji package missed them
        # However, emoji package is quite comprehensive. 
        # The prompt says: "Remove all emojis and decorative Unicode symbols"
        # We will trust `emoji.replace_emoji` for emojis.
        # Let's save it back.
        
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(cleaned_content)
            print(f"Cleaned emojis from {filepath}")
        except Exception as e:
            print(f"Failed to write to {filepath}: {e}")

def main():
    root_dir = r"c:\Users\HP\OneDrive\Desktop\FinalYearProject"
    
    # Extensions to process
    valid_extensions = {'.py', '.html', '.js', '.css', '.json', '.txt', '.md', '.csv', '.env'}
    
    # Directories to skip
    skip_dirs = {'__pycache__', 'venv', 'node_modules', '.git', '.idea', '.vscode'}

    for root, dirs, files in os.walk(root_dir):
        # Modify dirs in-place to skip
        dirs[:] = [d for d in dirs if d not in skip_dirs]
        
        for file in files:
            ext = os.path.splitext(file)[1].lower()
            if ext in valid_extensions:
                filepath = os.path.join(root, file)
                remove_emojis_from_file(filepath)

if __name__ == "__main__":
    main()
