import json
import os
import argparse


def process_json(input_file, output_file, data_path):
    # Read the original JSON file
    with open(input_file, 'r') as f:
        annotations = json.load(f)

    original_key_len = len(annotations)

    # Process the JSON data
    for key, value in list(annotations.items()):
        full_path = os.path.join(data_path, value["path"]+".tensor")
        if not os.path.exists(full_path):
            # If "path" exists but the file does not exist, remove the entry
            print(f"File does not exist: {full_path}")
            del annotations[key]

    # Save the processed JSON to a new file
    with open(output_file, 'w') as f:
        json.dump(annotations, f, indent=2)

    # Count the difference in keys between the original and processed JSON
    print(f"Original JSON has {original_key_len} entries")
    print(f"Processed JSON has {len(annotations)} entries")

    print(f"Processed JSON saved as {output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Process JSON file and remove entries where 'path' file does not exist.")
    parser.add_argument("--input_file", type=str, default="annotations/humanml3d/annotations_original.json")
    parser.add_argument("--output_file", type=str, default="annotations/humanml3d/annotations_processed.json")
    parser.add_argument("--data_path", type=str, default="../datasets/guoh3dfeats")

    args = parser.parse_args()

    process_json(args.input_file, args.output_file, args.data_path)
