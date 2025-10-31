"""
Training monitoring and visualization utilities
"""
import json
import os
import matplotlib.pyplot as plt
from typing import Dict, List

class TrainingMonitor:
    """Monitor and visualize training progress"""
    
    def __init__(self, log_dir: str = "logs"):
        """Initialize monitor"""
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
    
    def plot_training_history(self, history_path: str, output_path: str = None):
        """
        Plot training history from saved JSON file
        
        Args:
            history_path: Path to training_history.json
            output_path: Path to save plot (optional)
        """
        # Load history
        with open(history_path, 'r') as f:
            history = json.load(f)
        
        epochs = history['epochs']
        train_loss = history['train_loss']
        val_loss = history['val_loss']
        
        # Create plot
        plt.figure(figsize=(12, 6))
        
        plt.plot(epochs, train_loss, label='Training Loss', linewidth=2)
        plt.plot(epochs, val_loss, label='Validation Loss', linewidth=2)
        
        plt.xlabel('Epoch', fontsize=12)
        plt.ylabel('Loss', fontsize=12)
        plt.title('Training Progress', fontsize=14, fontweight='bold')
        plt.legend(fontsize=10)
        plt.grid(True, alpha=0.3)
        
        # Find best epoch
        best_epoch = epochs[val_loss.index(min(val_loss))]
        best_val_loss = min(val_loss)
        
        plt.axvline(x=best_epoch, color='r', linestyle='--', alpha=0.5, 
                   label=f'Best Epoch: {best_epoch}')
        
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            print(f"Plot saved to {output_path}")
        else:
            plt.show()
    
    def plot_rl_rewards(self, rewards: List[float], output_path: str = None):
        """
        Plot RL training rewards
        
        Args:
            rewards: List of episode rewards
            output_path: Path to save plot
        """
        plt.figure(figsize=(12, 6))
        
        plt.plot(rewards, alpha=0.6, linewidth=1)
        
        # Add moving average
        window = min(100, len(rewards) // 10)
        if window > 1:
            moving_avg = [sum(rewards[max(0, i-window):i+1]) / len(rewards[max(0, i-window):i+1]) 
                         for i in range(len(rewards))]
            plt.plot(moving_avg, label=f'Moving Average ({window} episodes)', 
                    linewidth=2, color='red')
        
        plt.xlabel('Episode', fontsize=12)
        plt.ylabel('Reward', fontsize=12)
        plt.title('RL Training Progress', fontsize=14, fontweight='bold')
        plt.legend(fontsize=10)
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            print(f"Plot saved to {output_path}")
        else:
            plt.show()
    
    def generate_report(self, checkpoint_dir: str, output_path: str = None):
        """
        Generate training report
        
        Args:
            checkpoint_dir: Directory containing checkpoints
            output_path: Path to save report
        """
        report = []
        report.append("=" * 60)
        report.append("TRAINING REPORT")
        report.append("=" * 60)
        report.append("")
        
        # Check for DL checkpoints
        dl_checkpoint_dir = os.path.join(checkpoint_dir, 'deep_learning')
        if os.path.exists(dl_checkpoint_dir):
            report.append("Deep Learning Training:")
            
            history_file = os.path.join(dl_checkpoint_dir, 'training_history.json')
            if os.path.exists(history_file):
                with open(history_file, 'r') as f:
                    history = json.load(f)
                
                report.append(f"  Total Epochs: {len(history['epochs'])}")
                report.append(f"  Best Val Loss: {min(history['val_loss']):.6f}")
                report.append(f"  Final Train Loss: {history['train_loss'][-1]:.6f}")
                report.append(f"  Final Val Loss: {history['val_loss'][-1]:.6f}")
            
            report.append("")
        
        # Check for RL checkpoints
        rl_checkpoint_dir = os.path.join(checkpoint_dir, 'reinforcement_learning')
        if os.path.exists(rl_checkpoint_dir):
            report.append("Reinforcement Learning Training:")
            
            checkpoints = [f for f in os.listdir(rl_checkpoint_dir) if f.endswith('.zip')]
            report.append(f"  Total Checkpoints: {len(checkpoints)}")
            
            if checkpoints:
                report.append(f"  Latest: {sorted(checkpoints)[-1]}")
            
            report.append("")
        
        report.append("=" * 60)
        
        report_text = "\n".join(report)
        
        if output_path:
            with open(output_path, 'w') as f:
                f.write(report_text)
            print(f"Report saved to {output_path}")
        else:
            print(report_text)
        
        return report_text

def create_dashboard_html(checkpoint_dir: str, output_path: str = "training_dashboard.html"):
    """
    Create HTML dashboard for training results
    
    Args:
        checkpoint_dir: Directory containing checkpoints
        output_path: Path to save HTML file
    """
    html = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Training Dashboard</title>
        <style>
            body {
                font-family: Arial, sans-serif;
                margin: 20px;
                background-color: #f5f5f5;
            }
            .container {
                max-width: 1200px;
                margin: 0 auto;
                background-color: white;
                padding: 30px;
                border-radius: 10px;
                box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            }
            h1 {
                color: #333;
                border-bottom: 3px solid #4CAF50;
                padding-bottom: 10px;
            }
            .metric {
                display: inline-block;
                margin: 15px;
                padding: 20px;
                background-color: #f9f9f9;
                border-radius: 8px;
                min-width: 200px;
            }
            .metric-value {
                font-size: 32px;
                font-weight: bold;
                color: #4CAF50;
            }
            .metric-label {
                color: #666;
                font-size: 14px;
            }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>🤖 Bitcoin Trading ML - Training Dashboard</h1>
            <p>Training results and model performance metrics</p>
            
            <h2>Deep Learning</h2>
            <div class="metrics">
                <div class="metric">
                    <div class="metric-label">Model Type</div>
                    <div class="metric-value">LSTM</div>
                </div>
                <div class="metric">
                    <div class="metric-label">Status</div>
                    <div class="metric-value">✅</div>
                </div>
            </div>
            
            <h2>Reinforcement Learning</h2>
            <div class="metrics">
                <div class="metric">
                    <div class="metric-label">Algorithm</div>
                    <div class="metric-value">PPO</div>
                </div>
                <div class="metric">
                    <div class="metric-label">Status</div>
                    <div class="metric-value">✅</div>
                </div>
            </div>
            
            <h2>Training Logs</h2>
            <p>Check the logs directory for detailed training information.</p>
        </div>
    </body>
    </html>
    """
    
    with open(output_path, 'w') as f:
        f.write(html)
    
    print(f"Dashboard saved to {output_path}")

if __name__ == "__main__":
    # Example usage
    monitor = TrainingMonitor()
    
    # Generate report
    monitor.generate_report('checkpoints')
    
    # Create dashboard
    create_dashboard_html('checkpoints')
