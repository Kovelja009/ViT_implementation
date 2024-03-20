import json

def save_params_to_json(filename, **kwargs):
    data_to_save = {
        "config": kwargs.get("config", {}),
        "metrics": kwargs.get("metrics", {})
    }
        
    with open(filename, "w") as f:
        json.dump(data_to_save, f, indent=4)

def read_config(filename):
    with open(filename, 'r') as f:
        config = json.load(f)
    return config