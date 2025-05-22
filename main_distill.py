import argparse
from omegaconf import OmegaConf
from trainer import TrainerDistillDifIR as TrainerDistill, TrainerFaceRecon

# import warnings
# warnings.filterwarnings("ignore")


def get_parser(**parser_kwargs):
    parser = argparse.ArgumentParser(**parser_kwargs)
    parser.add_argument(
            "--save_dir",
            type=str,
            default="./saved_logs",
            help="Folder to save the checkpoints and training log",
            )
    parser.add_argument(
            "--resume",
            type=str,
            const=True,
            default="",
            nargs="?",
            help="Resume from the save_dir or checkpoint",
            )
    parser.add_argument(
            "--cfg_path",
            type=str,
            default="./configs/realsr_swinunet_realesrgan256.yaml",
            help="Configs of yaml file",
            )
    parser.add_argument(
            "--steps",
            type=int,
            default=15,
            help="Hyper-parameters of diffusion steps",
            )
    parser.add_argument(
            "--alpha",
            type=float,
            default=0.5,
            help="Hyper-parameters of balance the losses (loss = (1-alpha)*loss_sr + alpha*loss_mica)",
            )
    args = parser.parse_args()

    return args

if __name__ == "__main__":
    args = get_parser()

    configs = OmegaConf.load(args.cfg_path)
    configs.diffusion.params.steps = args.steps

    #!!! fix for mica
    configs.MICA.output_dir = args.save_dir
    configs.alpha = args.alpha

    # merge args to config
    for key in vars(args):
        if key in ['cfg_path', 'save_dir', 'resume', ]:
            configs[key] = getattr(args, key)

    trainer = TrainerFaceRecon(configs) # TrainerDistill
    trainer.train()

# export OMP_NUM_THREADS=1
# python3 -m debugpy --listen 0.0.0.0:5678 --wait-for-client -m torch.distributed.run --nproc_per_node=3 main_distill.py --cfg_path configs/SinSR_vggface2.yaml --save_dir logs/SinSR
# python3 -m torch.distributed.run --nproc_per_node=3 main_distill.py --cfg_path configs/SinSR_vggface2.yaml --save_dir logs/SinSR