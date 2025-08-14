import json
import re
import os


def process_json_file(file_path):
    # Read JSON file
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except Exception as e:
        print(f"Error reading file {file_path}: {e}")
        return

    # Check JSON structure
    if 'videos' not in data:
        print(f"Invalid JSON structure: 'videos' key not found in {file_path}")
        return

    modified = False

    # Iterate through all videos
    for video in data['videos']:
        # Check if sample_type is person1, person2 or both
        if video.get('sample_type') not in ['person1', 'person2', 'both']:
            continue

        # Iterate through all Q&A pairs
        for qa_pair in video.get('qa_pairs', []):
            question = qa_pair.get('question', '')
            answer = qa_pair.get('answer', '')

            # Check the number of <> tags in the question
            entities = re.findall(r'<[^>]+>', question)

            # If there's only one entity and no <> tags in the answer, perform replacement
            if len(entities) == 1 and '<' not in answer:
                entity = entities[0]  # For example <Nz>

                # Check if pronouns need to be replaced
                # Replace "the man", "he", "He", "him", "Him" etc.
                patterns = [
                    (r'\bthe man\b', entity),
                    (r'\bThe man\b', entity),
                    (r'\bhe\b', entity),
                    (r'\bHe\b', entity),
                    (r'\bhim\b', entity),
                    (r'\bHim\b', entity),
                    (r'\bhis\b', entity),
                    (r'\bHis\b', entity)
                ]

                # Apply replacement
                new_answer = answer
                for pattern, replacement in patterns:
                    new_answer = re.sub(pattern, replacement, new_answer)

                # If answer was modified, update qa_pair
                if new_answer != answer:
                    qa_pair['answer'] = new_answer
                    modified = True

    # If there are modifications, write back to file
    if modified:
        # Create backup
        backup_path = file_path + '.bak'
        try:
            os.rename(file_path, backup_path)
            print(f"Created backup at {backup_path}")
        except Exception as e:
            print(f"Warning: Could not create backup: {e}")

        # Write modified file
        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            print(f"Successfully updated {file_path}")
        except Exception as e:
            print(f"Error writing to {file_path}: {e}")
            # Try to restore backup
            if os.path.exists(backup_path):
                try:
                    os.rename(backup_path, file_path)
                    print(f"Restored original file from backup")
                except:
                    print(f"Warning: Could not restore backup")
    else:
        print(f"No changes made to {file_path}")


if __name__ == "__main__":
    # Specify file path
    file_path = "/root/autodl-tmp/yufei/InternVideo/InternVideo2/multi_modality/<Nz>_<Ab>_test.json"

    # Replace placeholders in filename with actual values
    # If you need to actually replace these values, modify the code below
    # For example: file_path = file_path.replace('<Nz>', 'Nz3').replace('<Ab>', 'Ab2')

    # Process file
    process_json_file(file_path)