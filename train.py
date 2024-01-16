"""Cream
Copyright (c) 2023-present NAVER Cloud Corp.
MIT license"""
import argparse
import datetime
import os
import sys
from os.path import basename
from pathlib import Path

import pytorch_lightning as pl
import torch
from lightning_fabric.utilities.seed import seed_everything
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers.tensorboard import TensorBoardLogger
from pytorch_lightning.strategies.ddp import DDPStrategy
from sconf import Config

sys.path.insert(0, ".")
from lightning_module import CreamDataPLModule, CreamModelPLModule


def train(config):
    seed_everything(config.get("seed", 42), workers=False)

    if config.get("pretrained_checkpoint_path", None):
        print(f"load ckpt: {config.pretrained_checkpoint_path}")
        model_module = CreamModelPLModule.load_from_checkpoint(
            config.pretrained_checkpoint_path, config=config, strict=False
        )
    else:
        model_module = CreamModelPLModule(config)
    if config.get("llm_integration_enabled", False):
        for param in model_module.llm_backbone.parameters():
            param.requires_grad = False
        model_module.llm_backbone.eval()
    if config.get("freeze", None):
        if config.freeze.get("image_encoder", False):
            print("this will freeze image encoder")
            for param in model_module.model.image_encoder.parameters():
                param.requires_grad = False
        if config.freeze.get("aux_encoder", False):
            print("this will freeze text encoder")
            for param in model_module.model.aux_encoder.parameters():
                param.requires_grad = False
        if config.freeze.get("text_decoder", False):
            print("this will freeze text decoder")
            for param in model_module.model.text_decoder.parameters():
                param.requires_grad = False
        if config.freeze.get("aux_encoder_layer_only", False):
            print("this will freeze aux_encoder_layer_only")
            for param in model_module.model.aux_encoder.encoder.layers.parameters():
                param.requires_grad = False
        if config.freeze.get("proj", False):
            print("this will freeze proj")
            for param in model_module.proj.parameters():
                param.requires_grad = False

    if config.get("train_llm_with_lora", False):
        lora_target_regex = config.get("lora_target_regex", None)
        lora_rank = config.get("lora_rank", 16)
        print(f"train_llm_with_lora is set True, lora_rank: {lora_rank}, lora_target_regex: {lora_target_regex}")
        from peft import LoraConfig, get_peft_model

        model_module.llm_backbone = get_peft_model(
            model_module.llm_backbone,
            LoraConfig(
                r=lora_rank,
                lora_alpha=32,
                inference_mode=False,
                target_modules=lora_target_regex if lora_target_regex else None,
                lora_dropout=0.05,
                task_type="CAUSAL_LM",
            ),
        )
        model_module.llm_backbone.print_trainable_parameters()

    data_module = CreamDataPLModule(config)
    data_module.build_datasets(
        model=model_module.model,
        llm_tokenizer=model_module.llm_tokenizer if config.get("llm_integration_enabled", False) else None,
    )

    # logger = None # 이거랑 tokenizer 쪽 하드코딩
    # lr_callback = None
    logger = TensorBoardLogger(
        save_dir=config.get("result_path", "./"),
        name=config.exp_name,
        version=config.exp_version,
        default_hp_metric=False,
    )
    lr_callback = LearningRateMonitor(logging_interval="step")

    checkpoint_callback = ModelCheckpoint(
        dirpath=Path(config.get("result_path", "./")) / config.exp_name / config.exp_version,
        filename=config.callbacks.model_checkpoint.filename,
        every_n_epochs=config.callbacks.model_checkpoint.every_n_epochs,
        save_top_k=config.callbacks.model_checkpoint.save_top_k,
        save_last=config.callbacks.model_checkpoint.save_last,
        monitor=config.callbacks.monitor.target,
        mode=config.callbacks.monitor.mode,
    )

    max_epochs = 0
    reload_dataloaders_every_n_epochs = config.get("reload_dataloaders_every_n_epochs", 0)
    if len(config["train"].items()) > 1:
        reload_dataloaders_every_n_epochs = 1
    if len(config["train"].items()) > 10:
        raise ValueError(f"[Warning] Too many phases.")
    for _, phase_setting in config["train"].items():
        max_epochs += phase_setting["num_epochs"]

    print(f"Max epochs: {max_epochs}")

    ddp_strategy = DDPStrategy(find_unused_parameters=True)

    trainer = pl.Trainer(
        num_nodes=config.get("num_nodes", 1),
        reload_dataloaders_every_n_epochs=reload_dataloaders_every_n_epochs,
        strategy=ddp_strategy,
        accelerator=config.get("accelerator", "gpu"),
        devices=config.num_gpus_per_node if config.get("accelerator", "gpu") == "gpu" else 1,
        accumulate_grad_batches=config.get("accumulate_grad_batches", 1),
        max_epochs=max_epochs,
        check_val_every_n_epoch=config.callbacks.model_checkpoint.every_n_epochs
        if isinstance(config.callbacks.model_checkpoint.every_n_epochs, int)
        else 1,
        gradient_clip_val=config.get("gradient_clip_val", 1.0),
        precision=16 if config.get("accelerator", "gpu") == "gpu" else "bf16",
        num_sanity_val_steps=config.get("num_sanity_val_steps", 2),
        logger=logger,
        callbacks=[lr_callback, checkpoint_callback],
        limit_val_batches=config.get("limit_val_batches", 1.0),
    )

    trainer.fit(
        model_module,
        data_module,
        ckpt_path=config.get("resume_from_checkpoint_path", None),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--exp_version", type=str, required=False)
    args, left_argv = parser.parse_known_args()

    config = Config(args.config)
    config.argv_update(left_argv)

    if "exp_name" not in config:
        config.exp_name = basename(args.config).split(".")[0]
    config.exp_version = datetime.datetime.now().strftime("%Y%m%d_%H%M%S") if not args.exp_version else args.exp_version
    config.num_gpus_per_node = torch.cuda.device_count() if config.get("accelerator", "gpu") == "gpu" else 1

    dataset_path = config.get("dataset_path", None)
    if dataset_path is not None and (Path(dataset_path).exists() and Path(dataset_path).is_dir()):
        # train dataset_name_or_paths
        for phase_name in config.train.keys():
            dataset_name_or_paths = list()
            for dataset_name_or_path in config.train[phase_name]["dataset_name_or_paths"]:
                if not Path(dataset_name_or_path).exists():
                    dataset_name_or_path = os.path.join(dataset_path, dataset_name_or_path)
                dataset_name_or_paths.append(dataset_name_or_path)
            config.train[phase_name]["dataset_name_or_paths"] = dataset_name_or_paths
        # val dataset_name_or_paths
        dataset_name_or_paths = list()
        for dataset_name_or_path in config.val["dataset_name_or_paths"]:
            if not Path(dataset_name_or_path).exists():
                dataset_name_or_path = os.path.join(dataset_path, dataset_name_or_path)
            dataset_name_or_paths.append(dataset_name_or_path)
        config.val["dataset_name_or_paths"] = dataset_name_or_paths

    train(config)
