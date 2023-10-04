
from tqdm.auto import tqdm
import torch
from typing import Dict, List, Tuple
from data_loader import load_data
from model_builder import model_build
import matplotlib.pyplot as plt

# test_step
def test_step(model: torch.nn.Module, 
              dataloader: torch.utils.data.DataLoader, 
              loss_fn: torch.nn.Module,
              device: torch.device) -> Tuple[float, float]:
    
    model.eval()

    test_loss, test_acc = 0,0

    with torch.inference_mode():
        for batch, (x,y) in enumerate (dataloader):
            x,y=x.to(device), y.to(device)

            # 1. Forward pass
            test_pred_logits = model(x)

            # 2. Calculate and accumulate loss
            loss = loss_fn(test_pred_logits, y)
            test_loss += loss.item()
          
            # Calculate and accumulate accuracy
            test_pred_labels = test_pred_logits.argmax(dim=1)
            test_acc += ((test_pred_labels == y).sum().item()/len(test_pred_labels))
          
    # Adjust metrics to get average loss and accuracy per batch 
    test_loss = test_loss / len(dataloader)
    test_acc = test_acc / len(dataloader)
    return test_loss, test_acc