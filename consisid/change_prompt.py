import json
import copy


def add_reference_text(data, gender):
    """
    Recursively traverse JSON data and add reference image text after the deepest text description
    """
    reference_text = {
        "male": " He has the same facial structure and skin color as the reference image.",
        "female": " She has the same facial structure and skin color as the reference image."
    }

    if isinstance(data, dict):
        modified_dict = {}
        for key, value in data.items():
            if isinstance(value, str):
                # If it's a string (deepest description), add reference text
                modified_dict[key] = value + reference_text[gender]
            else:
                # If it's a dictionary, continue recursively
                modified_dict[key] = add_reference_text(value, gender)
        return modified_dict
    return data


def process_json_file(input_path, output_path):
    """
    Process JSON file and add reference image text
    """
    # Read original JSON file
    with open(input_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # Create deep copy of data
    modified_data = copy.deepcopy(data)

    # Process male part
    if 'male' in modified_data:
        modified_data['male'] = add_reference_text(modified_data['male'], 'male')

    # Process female part
    if 'female' in modified_data:
        modified_data['female'] = add_reference_text(modified_data['female'], 'female')

    # Save modified JSON file
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(modified_data, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    input_file = "/root/autodl-tmp/yufei/ConsisID/class_data.json"
    output_file = "/root/autodl-tmp/yufei/ConsisID/class_data_modified.json"

    try:
        process_json_file(input_file, output_file)
        print(f"Successfully processed JSON file.")
        print(f"Modified file saved to: {output_file}")
    except Exception as e:
        print(f"Error processing file: {str(e)}")