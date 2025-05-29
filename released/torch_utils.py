import torch

# Adapted from https://debuggercafe.com/saving-and-loading-the-best-model-in-pytorch/
class SaveBestModel:
    """
    Class to save the best model while training. If the current epoch's 
    validation loss is less than the previous least less, then save the
    model state.
    """
    def __init__(
        self, best_valid_auc=float(0)
    ):
        self.best_valid_auc = best_valid_auc
        
    def __call__(
        self, current_valid_auc, 
        epoch, model, optimizer, path
    ):
        if self.best_valid_auc < current_valid_auc :
            self.best_valid_auc = current_valid_auc
            # print(f"\nBest validation auc: {self.best_valid_auc}")
            # print(f"\nSaving best model for epoch: {epoch+1}\n")
            torch.save((model.state_dict(), optimizer.state_dict()), path)
            # torch.save({
            #     'model_state_dict': model.state_dict(),
            #     'optimizer_state_dict': optimizer.state_dict(),
            #     }, path)