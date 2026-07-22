import matplotlib.pyplot as plt
import seaborn as sns
import mlflow
import pandas as pd
from src.utils.config import config

def generate_plots():
    mlflow.set_tracking_uri(config.mlflow_tracking_uri)
    experiment = mlflow.get_experiment_by_name(config.mlflow_experiment_name)
    
    if not experiment:
        print(f"Experiment {config.mlflow_experiment_name} not found.")
        return
        
    runs = mlflow.search_runs(experiment_ids=[experiment.experiment_id])
    if runs.empty:
        print("No runs found.")
        return
        
    # Get the latest run ID
    latest_run_id = runs.iloc[0]["run_id"]
    client = mlflow.tracking.MlflowClient()
    
    metrics_to_fetch = ["train_loss", "val_loss", "train_acc", "val_acc"]
    metrics_data = {}
    
    for metric in metrics_to_fetch:
        history = client.get_metric_history(latest_run_id, metric)
        metrics_data[metric] = {m.step: m.value for m in history}
        
    # Create DataFrame
    df = pd.DataFrame(metrics_data)
    df.index.name = 'Epoch'
    df = df.dropna()
    
    sns.set_theme(style="whitegrid")
    
    # Plot Accuracy
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=df[['train_acc', 'val_acc']], linewidth=2.5)
    plt.title("Training vs Validation Accuracy", fontsize=16, pad=15)
    plt.xlabel("Epoch", fontsize=12)
    plt.ylabel("Accuracy", fontsize=12)
    plt.legend(["Training Accuracy", "Validation Accuracy"])
    plt.tight_layout()
    plt.savefig("accuracy_curve.png", dpi=300, bbox_inches='tight')
    print("Saved accuracy_curve.png")
    
    # Plot Loss
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=df[['train_loss', 'val_loss']], linewidth=2.5)
    plt.title("Training vs Validation Loss", fontsize=16, pad=15)
    plt.xlabel("Epoch", fontsize=12)
    plt.ylabel("Loss", fontsize=12)
    plt.legend(["Training Loss", "Validation Loss"])
    plt.tight_layout()
    plt.savefig("loss_curve.png", dpi=300, bbox_inches='tight')
    print("Saved loss_curve.png")

if __name__ == "__main__":
    generate_plots()
