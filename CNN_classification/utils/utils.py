import torch
from pathlib import Path

def save_model(model: torch.nn.Module,
               target_dir: str,
               model_name: str):
    
    # Create target directory
    target_dir_path=Path(target_dir)
    target_dir_path.mkdir(parents=True,
                        exist_ok=True)
    
    # Create model save path
    assert model_name.endswith(".pth") or model_name.endswith(".pt"), "model_name should end with '.pt' or '.pth'"
    model_save_path = target_dir_path / model_name

    # Save the model state_dict()
    print(f"[INFO] Saving model to: {model_save_path}")
    torch.save(obj=model.state_dict(),
             f=model_save_path)
    
def choose_device():
    if torch.cuda.is_available()==True:
        device = "cuda"
    else:
        device="cpu"
    return device
# cuda랑 cpu 이외의 다른 예들을 모르겠음




