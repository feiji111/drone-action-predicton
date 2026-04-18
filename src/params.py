import argparse


def parser_gen():
    parser = argparse.ArgumentParser(description="Train Multimodal Fusion Model")
    parser.add_argument("--data_dir", default="gdy")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--ckpt_dir", default="checkpoints_multimodal")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--warmup", type=int, default=350, help="warmup steps")
    parser.add_argument("--warmup_ratio", type=float, default=0.1)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--beta1", type=float, default=None, help="Adam beta 1.")
    parser.add_argument("--beta2", type=float, default=None, help="Adam beta 2.")
    parser.add_argument("--eps", type=float, default=None, help="Adam epsilon.")
    parser.add_argument("--wd", type=float, default=0.2)
    parser.add_argument("--device", default="cuda")

    parser.add_argument("--accum_freq", type=int, default=1)
    parser.add_argument("--distributed", action="store_true", default=False)

    # Model
    parser.add_argument("--backbone", type=str, default="resnet18")
    parser.add_argument(
        "--imu_hidden_dim", type=int, default=256, help="IMU hidden dim"
    )
    parser.add_argument("--imu_input_dim", type=int, default=22, help="IMU data dim")
    parser.add_argument(
        "--visual_hidden_dim", type=int, default=256, help="Visual hidden dim"
    )
    parser.add_argument(
        "--fusion_hidden_dim", type=int, default=256, help="Fusion hidden dim"
    )
    parser.add_argument("--num_classes", type=int, default=6)
    parser.add_argument(
        "--pretrained",
        action="store_true",
        default=False,
        help="Use pretrained visual backbone",
    )
    parser.add_argument(
        "--torchscript",
        default=False,
        action="store_true",
        help="torch.jit.script the model, also uses jit version of OpenAI models if pretrained=='openai'",
    )
    parser.add_argument(
        "--torchcompile",
        default=False,
        action="store_true",
        help="torch.compile() the model, requires pytorch 2.0 or later.",
    )

    # Profile
    parser.add_argument(
        "--profile",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Profile use torch.profiler",
    )

    # log and metric
    parser.add_argument(
        "--report_to",
        type=str,
        default="swanlab",
        help="Options are ['wandb', 'tensorboard', 'swanlab']",
    )
    parser.add_argument(
        "--debug", action="store_true", default=False, help="Set log level to debug"
    )
    parser.add_argument("--logs", type=str, default="./logs")
    parser.add_argument("--swanlab", action="store_true", help="Using swanlab")
    parser.add_argument(
        "--swanlab_project_name", type=str, default="Drone-action-estimate"
    )
    parser.add_argument("--name", type=str, default=None)
    parser.add_argument("--log_every_n_steps", type=int, default=10)

    args = parser.parse_args()

    args.world_size = 1
    args.swanlab = "swanlab" in args.report_to or "all" in args.report_to
    return args
