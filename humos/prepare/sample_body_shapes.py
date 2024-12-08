import torch
import json

def create_random_dict():
    # Set seeds for reproducibility
    torch.manual_seed(1234)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Create a dictionary with keys "14" and "96"
    random_dict = {
        "10": torch.randperm(10).tolist(),
        "14": torch.randperm(14).tolist(),
        "80": torch.randperm(80).tolist(),
        "96": torch.randperm(96).tolist()
    }

    return random_dict

def main():
    random_dict = create_random_dict()

    # Save the dictionary as JSON for each run
    with open(f"./stats/random_body_shapes.json", "w") as json_file:
        json.dump(random_dict, json_file)

if __name__ == "__main__":
    main()
