import os
import torch
from trainer import TrainerBase 


def main():
    # Paths
    dataset_dir = [
        '/users/ps1510/scratch/Programs/Sin3dFace/results/NoW_64_256_model1_s1_unkown_img/50000'
    ]
    results_dir = 'experiments/' + 'test_results'

    # Hyperparameters
    batch_size = 16
    val_size = 0.2  # 20% of the dataset for validation
    num_epochs = 20
    learning_rate = 1e-4
    mesh_feat_dim = 300
    fusion_hidden_dim = 256
    thresholds = {
        "median": 1.5,
        "mean": 1.5,
        "std": 1.5
    }
    save_log = 5  # save every 5 epochs
    resume = None  # Or path to model checkpoint

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Config dictionary to pass to TrainerBase
    configs = {
        'dataset_dir': dataset_dir,
        'results_dir': results_dir,
        'batch_size': batch_size,
        'val_size': val_size,
        'num_epochs': num_epochs,
        'learning_rate': learning_rate,
        'mesh_feat_dim': mesh_feat_dim,
        'fusion_hidden_dim': fusion_hidden_dim,
        'thresholds': thresholds,
        'save_log': save_log,
        'device': device,
        'resume': resume
    }

    trainer = TrainerBase(configs)
    trainer.run()

    # Optionally test or run inference on the same or different folders
    # trainer.test_loader(['/another/test/path'])
    # trainer.inference(['/another/test/path'])

if __name__ == '__main__':
    main()

# conda activate 3dSin_now
# cd /users/ps1510/scratch/Programs/Sin3dFace/predict_quality_now_benchmark
# python main.py 