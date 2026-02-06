import os
import torch
import heapq
from pathlib import Path

class CheckpointManager:
    """Manages model checkpoints, keeping only top K best models based on a metric."""
    
    def __init__(self, checkpoint_dir, max_keep=5, mode='max'):
        """
        Args:
            checkpoint_dir: Directory to save checkpoints
            max_keep: Maximum number of checkpoints to keep (default: 5)
            mode: 'max' for metrics where higher is better (accuracy),
                  'min' for metrics where lower is better (loss)
        """
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.max_keep = max_keep
        self.mode = mode
        
        # Heap to track top K checkpoints: (metric, epoch, filepath)
        # For 'max' mode: use negative values to simulate max-heap
        # For 'min' mode: use positive values (natural min-heap)
        self.checkpoints = []
    
    def save_checkpoint(self, model, optimizer, epoch, metric, extra_info=None):
        """
        Save a checkpoint and manage top K checkpoints.
        
        Args:
            model: PyTorch model
            optimizer: Optimizer
            epoch: Current epoch
            metric: Metric value to compare (e.g., accuracy, loss)
            extra_info: Dictionary of additional info to save
        
        Returns:
            bool: True if checkpoint was saved (in top K), False otherwise
        """
        # Determine if this checkpoint should be saved
        metric_value = -metric if self.mode == 'max' else metric
        
        should_save = (
            len(self.checkpoints) < self.max_keep or 
            metric_value < self.checkpoints[0][0]  # Better than worst in heap
        )
        
        if not should_save:
            print(f"Checkpoint not saved (not in top {self.max_keep})")
            return False
        
        # Create checkpoint filename
        checkpoint_name = f"model_epoch{epoch:03d}_acc{metric:.4f}.pth"
        checkpoint_path = self.checkpoint_dir / checkpoint_name
        
        # Save checkpoint
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'metric': metric,
        }
        
        if extra_info:
            checkpoint.update(extra_info)
        
        torch.save(checkpoint, checkpoint_path)
        print(f"✓ Saved checkpoint: {checkpoint_name} (metric: {metric:.4f})")
        
        # Add to heap
        heapq.heappush(self.checkpoints, (metric_value, epoch, str(checkpoint_path)))
        
        # Remove worst checkpoint if exceeding max_keep
        if len(self.checkpoints) > self.max_keep:
            worst_metric, worst_epoch, worst_path = heapq.heappop(self.checkpoints)
            if os.path.exists(worst_path):
                os.remove(worst_path)
                actual_metric = -worst_metric if self.mode == 'max' else worst_metric
                print(f"✗ Removed checkpoint: {Path(worst_path).name} (metric: {actual_metric:.4f})")
        
        return True
    
    def get_best_checkpoint(self):
        """Returns path to the best checkpoint."""
        if not self.checkpoints:
            return None
        
        # Best checkpoint is the one with smallest metric_value in heap
        # (which is the largest actual metric for 'max' mode)
        best = min(self.checkpoints, key=lambda x: x[0])
        return best[2]  # Return filepath
    
    def get_all_checkpoints(self):
        """Returns list of all saved checkpoints sorted by metric (best first)."""
        sorted_checkpoints = sorted(self.checkpoints, key=lambda x: x[0])
        return [(epoch, -metric if self.mode == 'max' else metric, path) 
                for metric, epoch, path in sorted_checkpoints]
    
    def load_checkpoint(self, model, optimizer=None, checkpoint_path=None):
        """
        Load a checkpoint.
        
        Args:
            model: PyTorch model
            optimizer: Optimizer (optional)
            checkpoint_path: Path to checkpoint (if None, loads best)
        
        Returns:
            dict: Checkpoint data
        """
        if checkpoint_path is None:
            checkpoint_path = self.get_best_checkpoint()
            if checkpoint_path is None:
                raise ValueError("No checkpoints available")
        
        print(f"Loading checkpoint: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, weights_only=False)
        
        model.load_state_dict(checkpoint['model_state_dict'])
        
        if optimizer and 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        return checkpoint