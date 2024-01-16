"""Cream
Copyright (c) 2023-present NAVER Cloud Corp.
MIT license"""
import bisect
import math
import sys
from itertools import chain
from pathlib import Path
from typing import List

import numpy as np
import pytorch_lightning as pl
import torch
from nltk import edit_distance
from pytorch_lightning.utilities import rank_zero_only
from torch import nn
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from transformers import LlamaForCausalLM, LlamaTokenizer

sys.path.insert(0, ".")
from cream import CreamConfig, CreamDataset, CreamModel, PrefixLMAttention


class CreamModelPLModule(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.cl_lamda = self.config.get("cl_lamda", 1.0)
        print(f"self.cl_lamda: {self.cl_lamda}")
        self.log_grad_norm = self.config.get("log_grad_norm", False)

        # set phases
        self.phase_names = list()
        self.phase_epochs = list()
        for phase_name, phase_setting in self.config["train"].items():
            self.phase_names.append(phase_name)
            self.phase_epochs.append(phase_setting["num_epochs"])
        self.phase_epochs = np.cumsum(self.phase_epochs).tolist()

        if self.config.get("pretrained_model_name_or_path", None):
            self.model = CreamModel.from_pretrained(self.config.pretrained_model_name_or_path)
        else:
            if self.config.get("pretrained_checkpoint_path", None):
                print(f"[Info] Load {config.pretrained_checkpoint_path}")
                self.model = CreamModel(
                    config=CreamConfig(
                        patch_size=self.config.get("patch_size", 14),
                        decoder_layers=self.config.decoder_layers,
                        encoder_layers=self.config.encoder_layers,
                        vision_layers=self.config.vision_layers,
                        max_patches=self.config.max_patches,
                        max_enc_position_embeddings=self.config.max_enc_position_embeddings,
                        max_position_embeddings=self.config.max_position_embeddings,
                        cl_enabled=self.config.get("cl_enabled", False),
                        num_aux_types=self.config.get("num_aux_types", 2),
                        llm_integration_enabled=self.config.get("llm_integration_enabled", False),
                        llm_vision_query_length=self.config.get("llm_vision_query_length", False),
                    ),
                    skip_custom_init=True,
                )
            else:
                skip_custom_init = self.config.get("skip_custom_init", False)
                self.model = CreamModel(
                    config=CreamConfig(
                        patch_size=self.config.get("patch_size", 14),
                        decoder_layers=self.config.get("decoder_layers", 12),
                        encoder_layers=self.config.get("encoder_layers", 12),
                        vision_layers=self.config.get("vision_layers", 18),
                        max_patches=self.config.get("max_patches", 3072),
                        max_enc_position_embeddings=self.config.get("max_enc_position_embeddings", 1024),
                        max_position_embeddings=self.config.get("max_position_embeddings", 128),
                        cl_enabled=self.config.get("cl_enabled", False),
                        num_aux_types=self.config.get("num_aux_types", 2),
                        llm_integration_enabled=self.config.get("llm_integration_enabled", False),
                        llm_vision_query_length=self.config.get("llm_vision_query_length", False),
                    ),
                    skip_custom_init=skip_custom_init,
                )
                print(f"[Info] skip_custom_init is set to {skip_custom_init}")

        if (
            "2" == torch.__version__[0]
            and not self.config.get("no_compile", False)
            and self.config.get("accelerator", "gpu") == "gpu"
        ):
            print("Train with a torch2.0 compiled Cream (PyTorch 2.0).")
            self.model = torch.compile(self.model)
            torch._dynamo.config.verbose = False
            torch._dynamo.config.suppress_errors = True

        if self.config.get("llm_integration_enabled", False):
            LLM_PATH = self.config.get("llm_dir_path", "~/vicuna-7b")

            self.llm_backbone = LlamaForCausalLM.from_pretrained(LLM_PATH, cache_dir=LLM_PATH)

            self.llm_util = PrefixLMAttention(prefix_length=self.config.get("llm_vision_query_length", 224))
            self.llm_backbone.model._prepare_decoder_attention_mask = (
                self.llm_util.prefixlm_llama_prepare_decoder_attention_mask
            )
            self.llm_tokenizer = LlamaTokenizer.from_pretrained(
                LLM_PATH,
                model_max_length=self.config.get("llm_max_length", 512),
            )

            self.llm_backbone.half().eval()
            self.proj = nn.Linear(self.model.text_decoder.config.d_model, self.llm_backbone.config.hidden_size)

            if "2" == torch.__version__[0] and not self.config.get("no_llm_compile", True):
                print("Train also with a torch2.0 compiled LLM (PyTorch 2.0).")
                self.llm_backbone = torch.compile(self.llm_backbone)
                torch._dynamo.config.verbose = False
                torch._dynamo.config.suppress_errors = True

    def on_train_epoch_start(self):
        phase_idx = bisect.bisect(self.phase_epochs, self.current_epoch)
        phase_name = self.phase_names[phase_idx] if phase_idx < len(self.phase_names) else self.phase_names[-1]
        optimizers = self.configure_optimizers(phase_name=phase_name)
        if len(optimizers) >= 2:
            self.lr_schedulers = optimizers[1]
        print(f"[Info] Current Phase : {phase_name}")
        for optimizer in self.trainer.optimizers:
            for param_group in optimizer.param_groups:
                if "lr" in self.config.train[phase_name]:
                    param_group["lr"] = self.config.train[phase_name]["lr"]
                    print(f"[Info] Updated LR: {self.config.train[phase_name]['lr']}")
                if "eps" in self.config.train[phase_name]:
                    param_group["eps"] = self.config.train[phase_name]["eps"]
                    print(f"[Info] Updated EPS: {self.config.train[phase_name]['eps']}")

    def training_step(self, batch, batch_idx):
        (
            input_ids,
            input_x_pos_ids,
            input_y_pos_ids,
            input_type_ids,
            attention_mask,
            image_input_tensors,
            image_attention_mask,
            decoder_input_ids,
            labels,
            cl_word_candidates_tensors,
            prompt_end_idxes,
        ) = (
            list(),
            list(),
            list(),
            list(),
            list(),
            list(),
            list(),
            list(),
            list(),
            list(),
            list(),
        )

        for batch_data in batch:
            input_ids.append(batch_data[0])
            input_x_pos_ids.append(batch_data[1])
            input_y_pos_ids.append(batch_data[2])
            input_type_ids.append(batch_data[3])
            attention_mask.append(batch_data[4])
            image_input_tensors.append(batch_data[5])
            image_attention_mask.append(batch_data[6])
            decoder_input_ids.append(batch_data[7])
            labels.append(batch_data[8])
            cl_word_candidates_tensors.append(batch_data[9])
            prompt_end_idxes.append(batch_data[10])
            is_textread = batch_data[11]
        input_ids = torch.cat(input_ids)
        input_x_pos_ids = torch.cat(input_x_pos_ids)
        input_y_pos_ids = torch.cat(input_y_pos_ids)
        input_type_ids = torch.cat(input_type_ids)
        attention_mask = torch.cat(attention_mask)
        image_input_tensors = torch.cat(image_input_tensors)
        image_attention_mask = torch.cat(image_attention_mask)
        decoder_input_ids = torch.cat(decoder_input_ids)
        labels = torch.cat(labels)
        cl_word_candidates_tensors = torch.cat(cl_word_candidates_tensors)
        prompt_end_idxes = torch.cat(prompt_end_idxes)

        if self.config.get("llm_integration_enabled", False):
            llm_input_ids = prompt_end_idxes
            llm_labels = labels
            output = self.model(
                input_ids=input_ids,
                input_x_pos_ids=input_x_pos_ids,
                input_y_pos_ids=input_y_pos_ids,
                input_type_ids=input_type_ids,
                attention_mask=attention_mask,
                image_input_tensors=image_input_tensors,
                image_attention_mask=image_attention_mask,
                decoder_input_ids=decoder_input_ids,
                cl_word_candidates=cl_word_candidates_tensors,
                output_hidden_states=True,
                # is_textread=is_textread,
            )
            inputs_embeds1 = self.proj(output.decoder_last_hidden_state[:, : self.llm_util.prefix_length])
            if self.config.get("train_llm_with_lora", False):
                inputs_embeds2 = self.llm_backbone.model.model.embed_tokens(llm_input_ids)
            else:
                inputs_embeds2 = self.llm_backbone.model.embed_tokens(llm_input_ids)
            inputs_embeds = torch.cat([inputs_embeds1, inputs_embeds2], dim=1)
            output2 = self.llm_backbone(
                inputs_embeds=inputs_embeds,
                attention_mask=torch.ones(
                    inputs_embeds.size()[:-1],
                    dtype=torch.long,
                    device=inputs_embeds.device,
                ),
                labels=llm_labels,
            )
            llm_loss = output2[0]
            self.log_dict(
                {
                    "train_loss": llm_loss + output.cl_loss,
                    "llm_loss": llm_loss,
                    "cl_loss": output.cl_loss,
                },
                sync_dist=True,
            )
            loss = llm_loss + output.cl_loss * self.cl_lamda
            return loss
        else:
            output = self.model(
                input_ids=input_ids,
                input_x_pos_ids=input_x_pos_ids,
                input_y_pos_ids=input_y_pos_ids,
                input_type_ids=input_type_ids,
                attention_mask=attention_mask,
                image_input_tensors=image_input_tensors,
                image_attention_mask=image_attention_mask,
                decoder_input_ids=decoder_input_ids,
                labels=labels,
                cl_word_candidates=cl_word_candidates_tensors,
                is_textread=is_textread,
            )
            self.log_dict(
                {
                    "train_loss": output.lm_loss + output.cl_loss,
                    "lm_loss": output.lm_loss,
                    "cl_loss": output.cl_loss,
                },
                sync_dist=True,
            )
            loss = output.lm_loss + output.cl_loss * self.cl_lamda
            return loss

    def validation_step(self, batch, batch_idx, dataset_idx=0):
        (
            input_ids,
            input_x_pos_ids,
            input_y_pos_ids,
            input_type_ids,
            attention_mask,
            image_input_tensors,
            image_attention_mask,
            decoder_input_ids,
            queries,
            decoder_label_ids,
            answers,
            dataset_idxes,
            levels,
            is_masked_text_visible,
            prompt_end_idxes,
            _,
        ) = batch
        dataset_idxes = dataset_idxes.detach().cpu().tolist()

        if levels[0] == "QA":
            gen_queries, predictions = list(), list()
            for i in range(len(input_ids)):
                if self.config.get("llm_integration_enabled", False):
                    llm_input_ids = prompt_end_idxes[i][: decoder_label_ids[i]].unsqueeze(0)
                    output = self.model(
                        input_ids=input_ids[i].unsqueeze(0),
                        input_x_pos_ids=input_x_pos_ids[i].unsqueeze(0),
                        input_y_pos_ids=input_y_pos_ids[i].unsqueeze(0),
                        input_type_ids=input_type_ids[i].unsqueeze(0),
                        attention_mask=attention_mask[i].unsqueeze(0),
                        image_input_tensors=image_input_tensors[i].unsqueeze(0),
                        image_attention_mask=image_attention_mask[i].unsqueeze(0),
                        decoder_input_ids=decoder_input_ids[i].unsqueeze(0),
                        output_hidden_states=True,
                    )
                    inputs_embeds1 = self.proj(output.decoder_last_hidden_state[:, : self.llm_util.prefix_length])
                    if self.config.get("train_llm_with_lora", False):
                        inputs_embeds2 = self.llm_backbone.model.model.embed_tokens(llm_input_ids)
                    else:
                        inputs_embeds2 = self.llm_backbone.model.embed_tokens(llm_input_ids)
                    inputs_embeds = torch.cat([inputs_embeds1, inputs_embeds2], dim=1)
                    gen_tokens = self.llm_backbone.generate(
                        inputs_embeds=inputs_embeds,
                        attention_mask=torch.ones(
                            inputs_embeds.size()[:-1],
                            dtype=torch.long,
                            device=inputs_embeds.device,
                        ),
                        pad_token_id=self.llm_tokenizer.pad_token_id,
                        eos_token_id=self.llm_tokenizer.eos_token_id,
                        use_cache=True,
                        bad_words_ids=[
                            [self.llm_tokenizer.unk_token_id],
                        ],
                        num_beams=1,
                        max_new_tokens=self.config.get("llm_max_length", 512) - inputs_embeds.size(1),
                        min_length=1,
                        repetition_penalty=1.5,
                        length_penalty=1,
                        early_stopping=True,
                    )
                    predictions.append(
                        self.llm_tokenizer.batch_decode(
                            gen_tokens[:, 1:] if gen_tokens[0, 0] == 0 else gen_tokens,
                            skip_special_tokens=True,
                            clean_up_tokenization_spaces=False,
                        )[0]
                        .replace(self.llm_tokenizer.pad_token, "")
                        .replace(self.llm_tokenizer.eos_token, "")
                        .replace("  ", " ")
                        .strip()
                    )
                    gen_queries.append(queries[i])
                else:
                    prediction = self.model.inference(
                        input_ids=input_ids[i].unsqueeze(0),
                        input_x_pos_ids=input_x_pos_ids[i].unsqueeze(0),
                        input_y_pos_ids=input_y_pos_ids[i].unsqueeze(0),
                        input_type_ids=input_type_ids[i].unsqueeze(0),
                        attention_mask=attention_mask[i].unsqueeze(0),
                        image_input_tensors=image_input_tensors[i].unsqueeze(0),
                        image_attention_mask=image_attention_mask[i].unsqueeze(0),
                        decoder_input_ids=decoder_input_ids[i][: prompt_end_idxes[i] + 1].unsqueeze(0),
                    )
                    gen_queries.append(prediction["queries"][0] if prediction["queries"][0] is not None else "")
                    predictions.append(prediction["predictions"][0] if prediction["predictions"][0] is not None else "")
        else:
            predictions = self.model.inference(
                input_ids=input_ids,
                input_x_pos_ids=input_x_pos_ids,
                input_y_pos_ids=input_y_pos_ids,
                input_type_ids=input_type_ids,
                attention_mask=attention_mask,
                image_input_tensors=image_input_tensors,
                image_attention_mask=image_attention_mask,
                decoder_input_ids=decoder_input_ids[:, : 3 + 1],
            )
            gen_queries = predictions["queries"]
            predictions = predictions["predictions"]

        scores = list()
        scores_not_case_sensitive = list()

        for idx, pred in enumerate(predictions):
            answer = answers[idx]
            scores.append(edit_distance(pred, answer) / max(len(pred), len(answer)))
            scores_not_case_sensitive.append(edit_distance(pred.lower(), answer.lower()) / max(len(pred), len(answer)))

        scores, scores_uncased, scores_anls = (
            list(),
            list(),
            list(),
        )
        scores_only, scores_only_uncased, scores_only_anls, scores_only_uncased_anls = (
            list(),
            list(),
            list(),
            list(),
        )
        for idx, (gen_query, prediction) in enumerate(zip(gen_queries, predictions)):
            query, gt = queries[idx], answers[idx]
            scores.append(edit_distance(prediction, gt) / max(len(prediction), len(gt)))

            if self.config.get("verbose", False) and len(scores) == 1:
                self.print(f"Query                 : {query}")
                self.print(f"Generated Query       : {gen_query}")
                self.print(f"GT                    : {gt}")
                self.print(f"Prediction            : {prediction}")
                self.print(f"Level                 : {levels[idx]}")
                self.print(f"Is masked text visible: {is_masked_text_visible[idx]}")
                self.print(f"Normed ED             : {scores[0]}")

                if self.config.get("llm_integration_enabled", False):
                    self.print(f"inputs_embeds1.size() : {inputs_embeds1.size()}")
                    self.print(f"inputs_embeds2.size() : {inputs_embeds2.size()}")
                    self.print(f"inputs_embeds.size()  : {inputs_embeds.size()}")
                    self.print(f"len(gen_tokens[0])       : {len(gen_tokens[0])}")
                    self.print(
                        f"self.llm_tokenizer.convert_ids_to_tokens(llm_input_ids[0]): {self.llm_tokenizer.convert_ids_to_tokens(llm_input_ids[0])}"
                    )
                    self.print(
                        f"self.llm_tokenizer.convert_ids_to_tokens(gen_tokens[0]): {self.llm_tokenizer.convert_ids_to_tokens(gen_tokens[0])}"
                    )
        return (
            scores,
            dataset_idxes,
        )

    def validation_epoch_end(self, validation_step_outputs):
        num_of_loaders = len(self.config.val.dataset_name_or_paths)
        cnt = [0] * num_of_loaders
        total_metric = [0] * num_of_loaders

        for validation_step_output in validation_step_outputs:
            for score, dataset_idx in zip(*validation_step_output):
                cnt[dataset_idx] += 1
                total_metric[dataset_idx] += score

        valid = False
        for dataset_idx in range(0, num_of_loaders):
            if cnt[dataset_idx] < 1:
                continue
            valid = True

            self.log_dict(
                {f"val_metric_{dataset_idx}th_dataset": total_metric[dataset_idx] / cnt[dataset_idx]},
                sync_dist=True,
            )
        if valid:
            self.log_dict(
                {
                    "val_metric": np.sum(total_metric) / np.sum(cnt),
                },
                sync_dist=True,
            )

    def configure_optimizers(self, phase_name: str = None):
        if phase_name is None:
            phase_name = self.phase_names[0]
        optimizer_config = self.config["train"][phase_name]

        if self.config.get("llm_integration_enabled", False):
            parameters = chain(self.model.parameters(), self.proj.parameters())
        else:
            parameters = self.model.parameters()
        optimizer = torch.optim.AdamW(
            parameters,
            lr=optimizer_config.get("lr", 1e-4),
            weight_decay=optimizer_config.get("decay", 1e-2),
            eps=optimizer_config.get("eps", 1e-6),
        )

        if optimizer_config.get("scheduler", None) == "cosine":
            max_iter = optimizer_config["synthetic_epoch_steps"] * optimizer_config["num_epochs"]
            scheduler = {
                "scheduler": self.cosine_scheduler(optimizer, max_iter, optimizer_config.get("warmup_steps", 0)),
                "name": "learning_rate",
                "interval": "step",
            }
            return [optimizer], [scheduler]
        else:
            return [optimizer]

    def lr_scheduler_step(self, scheduler, optimizer_idx, metric):
        if metric is None:
            scheduler.step()
        else:
            scheduler.step(metric)

    @staticmethod
    def cosine_scheduler(optimizer, training_steps, warmup_steps):
        def lr_lambda(current_step):
            if current_step < warmup_steps:
                return current_step / max(1, warmup_steps)
            progress = current_step - warmup_steps
            progress /= max(1, training_steps - warmup_steps)
            return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))

        return LambdaLR(optimizer, lr_lambda)

    def get_progress_bar_dict(self):
        items = super().get_progress_bar_dict()
        items.pop("v_num", None)
        items["exp_name"] = f"{self.config.get('exp_name', '')}"
        items["exp_version"] = f"{self.config.get('exp_version', '')}"
        return items

    @rank_zero_only
    def on_save_checkpoint(self, checkpoint):
        if self.config.get("llm_integration_enabled", False):
            checkpoint["state_dict"] = {k: v for k, v in checkpoint["state_dict"].items() if not "llm_backbone" in k}
        if self.config.callbacks.model_checkpoint.get("save_last_hf_format", False):
            save_path = Path(self.config.get("result_path", "./")) / self.config.exp_name / self.config.exp_version
            self.model.save_pretrained(save_path)
            self.model.tokenizer.save_pretrained(save_path)

    def on_load_checkpoint(self, checkpoint: dict) -> None:
        state_dict = checkpoint["state_dict"]
        model_state_dict = self.state_dict()
        for param_name in [
            "model._orig_mod.text_decoder.embed_positions.weight",
            "model._orig_mod.text_decoder.embed_positions.weight",
        ]:
            if (
                param_name in state_dict
                and param_name in model_state_dict
                and state_dict[param_name].size(0) != model_state_dict[param_name].size(0)
            ):
                print(f"ADJUST {param_name}")
                state_dict[param_name] = torch.nn.Parameter(
                    self.model.resize_bart_abs_pos_emb(
                        state_dict[param_name],
                        self.model.config.max_position_embeddings
                        + 2,  # https://github.com/huggingface/transformers/blob/v4.11.3/src/transformers/models/mbart/modeling_mbart.py#L118-L119
                    )
                )

    def optimizer_step(
        self,
        epoch,
        batch_idx,
        optimizer,
        optimizer_idx,
        optimizer_closure,
        on_tpu=False,
        using_lbfgs=False,
    ):
        phase_name = self.phase_names[0]
        optimizer_config = self.config["train"][phase_name]

        if optimizer_config.get("warmup_steps", 0) > 0:
            if self.trainer.global_step < optimizer_config.warmup_steps:
                lr_scale = max(
                    min(
                        1.0,
                        float(self.trainer.global_step + 1) / float(optimizer_config.warmup_steps),
                    ),
                    0.0001,
                )
                for pg in optimizer.param_groups:
                    pg["lr"] = lr_scale * optimizer_config.lr

        optimizer.step(closure=optimizer_closure)

        if self.log_grad_norm:
            log = {
                "grad_norm": get_grad_norm(
                    [
                        self.model,
                    ]
                ),
                "grad_norm_text_decoder": get_grad_norm(
                    [
                        self.model.text_decoder,
                    ]
                ),
            }
            if isinstance(self.model.image_encoder, nn.Module):
                log["grad_norm_image_encoder"] = get_grad_norm(
                    [
                        self.model.image_encoder,
                    ]
                )
            if isinstance(self.model.aux_encoder, nn.Module):
                log["grad_norm_aux_encoder"] = get_grad_norm(
                    [
                        self.model.aux_encoder,
                    ]
                )
            self.log_dict(log, sync_dist=True)


class CreamDataPLModule(pl.LightningDataModule):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.train_datasets = dict()
        self.val_datasets = list()

        self.phase_names = list()
        self.phase_epochs = list()
        for phase_name, phase_setting in self.config["train"].items():
            self.phase_names.append(phase_name)
            self.phase_epochs.append(phase_setting["num_epochs"])
        self.phase_epochs = np.cumsum(self.phase_epochs).tolist()

    def train_dataloader(self):
        phase_idx = bisect.bisect(self.phase_epochs, self.trainer.current_epoch)
        phase_name = self.phase_names[phase_idx] if phase_idx < len(self.phase_names) else self.phase_names[-1]
        print(f"[Info] Phase {phase_name} Start!")
        loaders = list()
        for dataset in self.train_datasets[phase_name]:
            dataset.set_new_epoch(self.trainer.current_epoch)
            loaders.append(
                DataLoader(
                    dataset,
                    batch_size=self.config["train"][phase_name]["batch_size"],
                    num_workers=self.config["train"][phase_name].get("num_workers", 0),
                    pin_memory=True,
                    shuffle=True,
                )
            )
        return loaders

    def val_dataloader(self):
        loaders = list()
        for dataset in self.val_datasets:
            loaders.append(
                DataLoader(
                    dataset,
                    batch_size=self.config["val"]["batch_size"],
                    num_workers=self.config["val"].get("num_workers", 0),
                    pin_memory=True,
                    shuffle=False,
                )
            )
        return loaders

    def build_datasets(self, model, llm_tokenizer=None):
        self.train_datasets = dict()
        last_valid_phase_name = None
        for phase_name, dataset_config in self.config["train"].items():
            print(f"[Info] The setting of phase {phase_name}: {dataset_config}")
            if dataset_config.get("dataset_name_or_paths", None):
                last_valid_phase_name = phase_name
                self.train_datasets[phase_name] = [
                    CreamDataset(
                        dataset_name_or_paths=dataset_config["dataset_name_or_paths"],
                        dataset_type=dataset_config["dataset_type"],
                        multiples=dataset_config.get(
                            "multiples",
                            [1] * len(dataset_config["dataset_name_or_paths"]),
                        ),
                        tasks=dataset_config.get(
                            "tasks",
                            ["qa"] * len(dataset_config["dataset_name_or_paths"]),
                        ),
                        model=model,
                        split="train",
                        textread_batch_ratio=dataset_config.get("textread_batch_ratio", 0.0),
                        textread_config=dataset_config.get("textread", None),
                        mlm_config=dataset_config.get("mlm", {}),
                        synthetic_epoch_num_samples=-1
                        if "synthetic_epoch_steps" not in dataset_config
                        else dataset_config["synthetic_epoch_steps"]
                        * dataset_config.get("batch_size", 1)
                        * self.config.get("accumulate_grad_batches", 1)
                        * self.config.get("num_gpus_per_node", 1)
                        * self.config.get("num_nodes", 1),
                        qa_config=dataset_config.get(
                            "qa", {"llm_integration_enabled": self.config.get("llm_integration_enabled", False)}
                        ),
                        llm_integration_enabled=self.config.get("llm_integration_enabled", False),
                        llm_tokenizer=llm_tokenizer,
                        llm_vision_query_length=self.config.get("llm_vision_query_length", 32),
                        llm_cream_query_length=self.config.get("llm_cream_query_length", 64),
                        llm_max_length=self.config.get("llm_max_length", 512),
                    ),
                ]
            else:
                self.train_datasets[phase_name] = self.train_datasets[last_valid_phase_name]

        self.val_datasets = list()
        self.val_datasets.append(
            CreamDataset(
                dataset_name_or_paths=self.config["val"]["dataset_name_or_paths"],
                multiples=self.config["val"].get(
                    "multiples",
                    [1] * len(self.config["val"]["dataset_name_or_paths"]),
                ),
                tasks=self.config["val"].get(
                    "tasks",
                    ["qa"] * len(self.config["val"]["dataset_name_or_paths"]),
                ),
                dataset_type=self.config["val"]["dataset_type"],
                model=model,
                split="validation",
                mlm_config=self.config["val"].get("mlm", {}),
                qa_config=self.config["val"].get("qa", {}),
                llm_integration_enabled=self.config.get("llm_integration_enabled", False),
                llm_tokenizer=llm_tokenizer,
                llm_vision_query_length=self.config.get("llm_vision_query_length", 32),
                llm_cream_query_length=self.config.get("llm_cream_query_length", 64),
                llm_max_length=self.config.get("llm_max_length", 512),
            )
        )


def get_grad_norm(models: List[nn.Module]):
    norm = 0
    for model in models:
        for p_name, p in model.named_parameters():
            try:
                norm += torch.linalg.norm(p.grad.detach().data).item() ** 2
            except:
                pass
    return norm**0.5
