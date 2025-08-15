# train.py

import argparse
from pathlib import Path
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger
from transformers import AutoTokenizer, AutoConfig, AutoModel
from datamodule import PersonalizedVideoDataModule
from lit_model import PersonalizedVideoLightningModule

def get_args():
    parser = argparse.ArgumentParser()
    # Model related
    parser.add_argument("--model_path", type=str,
                        default="/root/autodl-tmp/yufei/InternVideo/InternVideo2/multi_modality/InternVideo2_chat_8B_HD_finetune")
    parser.add_argument("--sks_name", type=str, default="<Bo>")
    parser.add_argument("--num_personal_token", type=int, default=16)

    # Training related
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--num_epochs", type=int, default=6)
    parser.add_argument("--personal_token_lr", type=float, default=1e-4)
    parser.add_argument("--token_lr", type=float, default=1e-5)
    parser.add_argument("--qformer_lr", type=float, default=1e-5)
    parser.add_argument("--lora_lr", type=float, default=1e-5)

    # Data related
    parser.add_argument("--data_root", type=str,
                        default="/root/autodl-tmp/yufei/InternVideo/InternVideo2/multi_modality")

    # Log related
    parser.add_argument("--log_dir", type=str, default="./logs")
    parser.add_argument("--save_dir", type=str, default="./checkpoints")
    parser.add_argument("--log_steps", type=int, default=10)
    parser.add_argument("--save_epochs", type=int, default=2)

    # Mode related
    parser.add_argument("--mode", type=str, default="train", choices=["train", "test"],
                        help="train: train and test; test: only test")
    parser.add_argument("--checkpoint_path", type=str, default=None,
                        help="Path to checkpoint for testing")

    args = parser.parse_args()
    return args

def main():
    args = get_args()

    # Define learning rates
    learning_rates = {
        'personal_token_lr': args.personal_token_lr,
        'token_lr': args.token_lr,
        'qformer_lr': args.qformer_lr,
        'lora_lr': args.lora_lr
    }

    # Define model configuration
    config_path = args.model_path
    config = AutoConfig.from_pretrained(config_path, trust_remote_code=True)
    reference_config_path = Path(config_path) / "config.json"

    # Automatically supplement missing fields (if needed)
    # Here assumes that `ensure_complete_config` function has been integrated into `LightningModule`
    # If needed, can load and process here

    # Initialize tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        config_path,
        trust_remote_code=True,
        use_fast=False
    )

    # Define prefix tokens
    prefix_tokens = [f'<token{i}>' for i in range(args.num_personal_token)]

    # Initialize LightningModule
    model = PersonalizedVideoLightningModule(
        config_path=config_path,
        sks_name=args.sks_name,
        prefix_tokens=prefix_tokens,
        learning_rates=learning_rates,
        model_config=config.to_dict()
    )

    # Initialize DataModule
    data_module = PersonalizedVideoDataModule(
        data_root=args.data_root,
        sks_name=args.sks_name,
        tokenizer=tokenizer,
        batch_size=args.batch_size,
        num_workers=4
    )

    # Set up callbacks
    checkpoint_callback = ModelCheckpoint(
        monitor='val_loss',
        dirpath=Path(args.save_dir) / args.sks_name,
        filename='{epoch}-{val_loss:.2f}',
        save_top_k=3,
        mode='min',
    )

    lr_monitor = LearningRateMonitor(logging_interval='step')

    # Set up logger
    logger = TensorBoardLogger(
        save_dir=args.log_dir,
        name=args.sks_name
    )

    # Initialize Trainer
    trainer = pl.Trainer(
        max_epochs=args.num_epochs,
        gpus=2 if torch.cuda.device_count() >=2 else 1,
        logger=logger,
        callbacks=[checkpoint_callback, lr_monitor],
        precision=16 if torch.cuda.is_available() else 32,  # Use mixed precision
        accelerator='gpu' if torch.cuda.is_available() else 'cpu',
        strategy='ddp' if torch.cuda.device_count() >1 else None
    )

    if args.mode == "train":
        # Start training
        trainer.fit(model, datamodule=data_module)

        # Run test
        trainer.test(model, datamodule=data_module)

    elif args.mode == "test":
        assert args.checkpoint_path is not None, "checkpoint_path must be specified in test mode"

        # Load checkpoint
        model = PersonalizedVideoLightningModule.load_from_checkpoint(
            checkpoint_path=args.checkpoint_path,
            config_path=config_path,
            sks_name=args.sks_name,
            prefix_tokens=prefix_tokens,
            learning_rates=learning_rates,
            model_config=config.to_dict()
        )

        # Load model weights
        model.model.load_state_dict(torch.load(Path(args.checkpoint_path) / 'pytorch_model.bin', map_location=torch.device('cuda' if torch.cuda.is_available() else 'cpu')))

        # Expand vocabulary
        tokenizer.add_tokens(model.sks_tokens + model.prefix_tokens)
        model.model.lm.resize_token_embeddings(len(tokenizer))

        # Set model to evaluation mode
        model.eval()

        # Initialize DataModule
        data_module.setup()

        # Run test
        trainer.test(model, datamodule=data_module)

if __name__ == "__main__":
    main()