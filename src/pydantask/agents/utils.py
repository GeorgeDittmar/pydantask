from pathlib import Path

def get_incremented_path(base_name: str, extension: str, directory: Path = Path("checkpoints")) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    
    counter = 1
    # Initial path attempt: checkpoints/state_1.json
    target_path = directory / f"{base_name}_{counter}.{extension}"
    
    # Keep incrementing until we find a filename that doesn't exist
    while target_path.exists():
        counter += 1
        target_path = directory / f"{base_name}_{counter}.{extension}"
        
    return target_path