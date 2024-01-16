"""Cream
Copyright (c) 2023-present NAVER Cloud Corp.
MIT license"""
import copy
import json
import os
import pickle
import random
import sys
from glob import glob
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, List, Tuple

import lmdb
import numpy as np
import torch
from PIL import Image, ImageDraw
from torch.utils.data import Dataset

from .model import CreamModel

Image.MAX_IMAGE_PIXELS = 1000000000
STOPWORDS = set("of and to in for a is that be on with by are as . or will from".split())


def rand_word_fn(
    words: List[Dict[str, Any]],
    n: int,
    alpha: float = 0.75,
    stopword_p: float = 0.9,
):
    counter = dict()
    text2idx = dict()
    for word_idx in sorted(range(len(words)), key=lambda x: random.random()):
        text = words[word_idx]["text"].strip()
        if len(text) < 1 or (np.random.rand() < stopword_p and text.lower() in STOPWORDS):
            continue
        if text not in counter:
            counter[text] = 0
            text2idx[text] = word_idx
        counter[text] += 1

    idxes = list()
    p = list()
    for k, v in counter.items():
        idxes.append(text2idx[k])
        p.append(v**alpha)
    p = np.array(p) / np.sum(p)

    n = min(n, len(idxes))
    if np.sum(p) != 1.0:
        p = np.ones(len(idxes)) * 1 / len(idxes)
    return np.random.choice(idxes, n, p=p, replace=False)


class CreamDataset(Dataset):
    def __init__(
        self,
        dataset_name_or_paths: list,
        model: CreamModel,
        multiples: list = [],
        tasks: list = [],
        dataset_type: str = "default",
        split: str = "train",
        textread_batch_ratio: float = 0.0,
        textread_config: Dict[str, Any] = None,
        mlm_config: Dict[str, Any] = dict(),
        ignore_id: int = -100,
        fix_seed_for_test: bool = False,
        synthetic_epoch_num_samples: int = -1,
        qa_config: Dict[str, Any] = dict(),
        max_patches: int = None,
        llm_integration_enabled: bool = False,
        llm_tokenizer=None,
        llm_vision_query_length=224,
        llm_cream_query_length=64,
        llm_max_length=512,
        custom_get_dataset_and_lengths=None,
        custom_get_image_and_meta_data=None,
        custom_get_random_question_generation_prompt=None,
        custom_get_random_question_answering_prompt=None,
        custom_get_random_captioning_prompt=None,
        custom_get_random_text_reading_prompt=None,
        custom_get_random_masked_text_prediction_prompt=None,
    ):
        super().__init__()

        self.dataset_name_or_paths = dataset_name_or_paths
        self.dataset_type = dataset_type

        self.get_dataset_and_lengths = custom_get_dataset_and_lengths
        self.get_image_and_meta_data = custom_get_image_and_meta_data
        if dataset_type == "default":
            self.get_dataset_and_lengths = (
                self._get_dataset_and_lengths_default
                if self.get_dataset_and_lengths is None
                else self.get_dataset_and_lengths
            )
            self.get_image_and_meta_data = (
                self._get_image_and_meta_data_default
                if self.get_image_and_meta_data is None
                else self.get_image_and_meta_data
            )
        elif dataset_type == "lmdb":
            self.get_dataset_and_lengths = (
                self._get_dataset_and_lengths_lmdb
                if self.get_dataset_and_lengths is None
                else self.get_dataset_and_lengths
            )
            self.get_image_and_meta_data = (
                self._get_image_and_meta_data_lmdb
                if self.get_image_and_meta_data is None
                else self.get_image_and_meta_data
            )
        else:
            if custom_get_dataset_and_lengths is None or custom_get_image_and_meta_data is None:
                raise ValueError(
                    f"The current dataset_type ({dataset_type}) is not supported."
                    + " Please set proper custom methods (`custom_get_dataset_and_lengths` and `custom_get_image_and_meta_data`)"
                    + " for the dataset class."
                )
            else:
                # custom methods are used
                pass
        self.datasets, self.dataset_lengths = self.get_dataset_and_lengths(dataset_name_or_paths, split)
        if len(multiples) != len(dataset_name_or_paths):
            self.multiples = [1] * len(dataset_name_or_paths)
        else:
            self.multiples = multiples
        if len(tasks) != len(dataset_name_or_paths):
            self.tasks = ["qa"] * len(dataset_name_or_paths)
        else:
            self.tasks = tasks
        self.model = model
        self.split = split

        self.textread_batch_ratio = textread_batch_ratio
        self.textread_config = textread_config
        self.mlm_config = dict() if mlm_config is None else mlm_config
        self.ignore_id = ignore_id
        self.qa_config = dict() if qa_config is None else qa_config
        self.num_aux_types = self.model.config.num_aux_types

        self.cumulative_dataset_lengths = []
        cum_dataset_length = 0
        for this_length, multi in zip(self.dataset_lengths, self.multiples):
            if split == "train":
                cum_dataset_length += this_length * multi
            else:
                cum_dataset_length += this_length
            self.cumulative_dataset_lengths.append(cum_dataset_length)

        self.synthetic_epoch_num_samples = synthetic_epoch_num_samples
        self.original_synthetic_epoch_ticks = None
        self.synthetic_epoch_ticks = None
        self.cum_dataset_length = cum_dataset_length
        if synthetic_epoch_num_samples > cum_dataset_length:
            raise Exception(
                f"synthetic_epoch_num_samples > cum_dataset_length : {synthetic_epoch_num_samples} > {cum_dataset_length}"
            )
        elif synthetic_epoch_num_samples < 1:
            synthetic_epoch_num_samples = cum_dataset_length
            self.original_synthetic_epoch_ticks = np.arange(cum_dataset_length, dtype=int)
            self.synthetic_epoch_ticks = self.original_synthetic_epoch_ticks + 0
        else:
            self.original_synthetic_epoch_ticks = np.linspace(
                0,
                cum_dataset_length,
                synthetic_epoch_num_samples,
                endpoint=False,
                dtype=int,
            )
            self.synthetic_epoch_ticks = self.original_synthetic_epoch_ticks + 0
        self.dataset_length = synthetic_epoch_num_samples
        print(f"[Info] Dataset Length: {self.dataset_length}")

        if llm_integration_enabled:
            assert llm_tokenizer is not None
            print("[Info] LLM Intergration is enabled.")
        self.llm_integration_enabled = llm_integration_enabled
        self.llm_tokenizer = llm_tokenizer
        self.llm_vision_query_length = llm_vision_query_length
        self.llm_cream_query_length = llm_cream_query_length
        self.llm_max_length = llm_max_length

        if custom_get_random_question_generation_prompt is not None:
            self.get_random_question_generation_prompt = custom_get_random_question_generation_prompt
        if custom_get_random_question_answering_prompt is not None:
            self.get_random_question_answering_prompt = custom_get_random_question_answering_prompt
        if custom_get_random_captioning_prompt is not None:
            self.get_random_captioning_prompt = custom_get_random_captioning_prompt
        if custom_get_random_text_reading_prompt is not None:
            self.get_random_text_reading_prompt = custom_get_random_text_reading_prompt
        if custom_get_random_masked_text_prediction_prompt is not None:
            self.get_random_masked_text_prediction_prompt = custom_get_random_masked_text_prediction_prompt

        self.fix_seed_for_test = fix_seed_for_test
        if fix_seed_for_test and self.split != "train":
            torch.manual_seed(0)
            random.seed(0)
            np.random.seed(0)

    @staticmethod
    def _get_dataset_and_lengths_default(dataset_name_or_paths, split):
        datasets = list()
        dataset_lengths = list()
        for dataset_name_or_path in dataset_name_or_paths:
            dataset = []
            images = sorted(glob(str(Path(dataset_name_or_path) / split / "image/*")))
            metas = sorted(glob(str(Path(dataset_name_or_path) / split / "meta/*")))
            assert len(images) == len(metas), print("There are a conflict between the number of image and meta data.")
            for img_path, meta_path in zip(images, metas):
                dataset.append(
                    {
                        "img_path": img_path,
                        "meta": json.load(open(meta_path)),
                    }
                )
            datasets.append(dataset)
            dataset_lengths.append(len(dataset))
        return datasets, dataset_lengths

    @staticmethod
    def _get_dataset_and_lengths_lmdb(dataset_name_or_paths, split):
        datasets = list()
        dataset_lengths = list()
        for dataset_name_or_path in dataset_name_or_paths:
            img_dataset = lmdb.open(
                str(Path(dataset_name_or_path) / split / "image"),
                max_readers=256,
                readonly=True,
                lock=False,
                readahead=False,
                meminit=False,
            )
            img_dataset_txn = img_dataset.begin(write=False)
            meta_dataset = lmdb.open(
                str(Path(dataset_name_or_path) / split / "meta"),
                max_readers=256,
                readonly=True,
                lock=False,
                readahead=False,
                meminit=False,
            )
            meta_dataset_txn = meta_dataset.begin(write=False)
            datasets.append(
                {
                    "img_datasets": img_dataset,
                    "img_dataset_txn": img_dataset_txn,
                    "meta_dataset": meta_dataset,
                    "meta_dataset_txn": meta_dataset_txn,
                }
            )
            dataset_lengths.append(int(meta_dataset_txn.get("num_data".encode("utf-8"))))
        return datasets, dataset_lengths

    def set_new_epoch(self, epoch=0) -> int:
        self.synthetic_epoch_ticks = self.original_synthetic_epoch_ticks + epoch
        self.synthetic_epoch_ticks = np.array(
            [
                idx if idx < self.cum_dataset_length else idx - self.cum_dataset_length
                for idx in self.synthetic_epoch_ticks
            ],
            dtype=int,
        )

    def __len__(self) -> int:
        return self.dataset_length

    def get_dataset_and_idx(self, idx):
        idx = self.synthetic_epoch_ticks[idx]
        for dataset_idx, cumulative_dataset_length in enumerate(self.cumulative_dataset_lengths):
            if dataset_idx == 0:
                if idx < cumulative_dataset_length:
                    return (
                        dataset_idx,
                        self.datasets[dataset_idx],
                        self.tasks[dataset_idx],
                        int(idx % self.dataset_lengths[dataset_idx]),
                    )
            else:
                if self.cumulative_dataset_lengths[dataset_idx - 1] <= idx and idx < cumulative_dataset_length:
                    return (
                        dataset_idx,
                        self.datasets[dataset_idx],
                        self.tasks[dataset_idx],
                        int(
                            (idx - self.cumulative_dataset_lengths[dataset_idx - 1]) % self.dataset_lengths[dataset_idx]
                        ),
                    )

        dataset_idx = -1
        return (
            dataset_idx,
            self.datasets[dataset_idx],
            self.tasks[dataset_idx],
            int((idx - self.cumulative_dataset_lengths[dataset_idx - 1]) % self.dataset_lengths[dataset_idx]),
        )

    def prepare_text_input(
        self,
        img_scale_info,
        x_scale,
        y_scale,
        meta,
        max_patches,
        masked_ids={},
        is_masked_text_visible=True,
        verbose=False,
        sample_words_for_cl=0,
    ):
        try:
            enc_input_ids = torch.tensor([], dtype=torch.long)
            enc_input_x_pos_ids = torch.tensor([], dtype=torch.long)
            enc_input_y_pos_ids = torch.tensor([], dtype=torch.long)
            enc_input_type_ids = torch.tensor([], dtype=torch.long)

            cl_word_candidates_tensor = -torch.ones(1024, 2)

            cum_len = 0
            word_start_idxes = []
            cl_word_candidates = []

            aux_boxes = [(meta["words"], 1)]
            if self.num_aux_types > 1 and "objects" in meta:
                aux_boxes.append((meta["objects"], 2))
            for items, input_type_id in aux_boxes:
                for word in items:
                    if word and word["text"].strip():
                        subwords = self.model.tokenizer(word["text"].strip(), add_special_tokens=False).input_ids

                        if len(subwords):
                            x0, y0 = word["bbox"][0]
                            x1, y1 = word["bbox"][1]
                            x2, y2 = word["bbox"][2]
                            x3, y3 = word["bbox"][3]

                            x0 = (x0 + x1 + x2 + x3) / 4
                            y0 = (y0 + y1 + y2 + y3) / 4

                            scaled_x0 = x0 * x_scale
                            scaled_y0 = y0 * y_scale
                            _x_pos_id = min(
                                max(scaled_x0 // img_scale_info["patch_width"], 0),
                                min(
                                    img_scale_info["num_feasible_cols"],
                                    max_patches,
                                    self.model.config.max_patches,
                                )
                                - 1,
                            )
                            x_pos_id = _x_pos_id + self.model.tokenizer.pad_token_id + 1
                            _y_pos_id = min(
                                max(scaled_y0 // img_scale_info["patch_height"], 0),
                                min(
                                    img_scale_info["num_feasible_rows"],
                                    max_patches,
                                    self.model.config.max_patches,
                                )
                                - 1,
                            )
                            y_pos_id = _y_pos_id + self.model.tokenizer.pad_token_id + 1

                            if input_type_id == 1 and word["id"] in masked_ids and not is_masked_text_visible:
                                mask_token_id = self.model.tokenizer.mask_token_id

                                subwords = [mask_token_id]
                                ids = torch.tensor([mask_token_id] * len(subwords), dtype=torch.long)
                            else:
                                ids = torch.tensor(subwords, dtype=torch.long)

                                cl_word_candidates.append(
                                    (
                                        cum_len,
                                        _y_pos_id * img_scale_info["num_feasible_cols"] + _x_pos_id,
                                    )
                                )

                            x_pos_ids = torch.tensor([x_pos_id] * len(subwords), dtype=torch.long)
                            y_pos_ids = torch.tensor([y_pos_id] * len(subwords), dtype=torch.long)
                            input_type_ids = torch.tensor([input_type_id] * len(subwords), dtype=torch.long)
                            word_start_idxes.append(cum_len)
                            enc_input_ids = torch.cat([enc_input_ids, ids])
                            enc_input_x_pos_ids = torch.cat([enc_input_x_pos_ids, x_pos_ids])
                            enc_input_y_pos_ids = torch.cat([enc_input_y_pos_ids, y_pos_ids])
                            enc_input_type_ids = torch.cat([enc_input_type_ids, input_type_ids])
                            cum_len += len(subwords)

            enc_attention_mask = torch.ones(self.model.config.max_enc_position_embeddings, dtype=torch.int)

            if cum_len < self.model.config.max_enc_position_embeddings:
                padding = torch.tensor(
                    [self.model.tokenizer.pad_token_id] * (self.model.config.max_enc_position_embeddings - cum_len)
                )

                enc_input_ids = torch.cat([enc_input_ids, padding])
                enc_input_x_pos_ids = torch.cat([enc_input_x_pos_ids, padding])
                enc_input_y_pos_ids = torch.cat([enc_input_y_pos_ids, padding])
                enc_input_type_ids = torch.cat(
                    [
                        enc_input_type_ids,
                        torch.tensor([0] * (self.model.config.max_enc_position_embeddings - cum_len)),
                    ]
                )
                enc_attention_mask[cum_len:] = 0
            elif cum_len > self.model.config.max_enc_position_embeddings:
                total_subword_idxes = list(range(0, cum_len))
                sampled_idxes = total_subword_idxes[: self.model.config.max_enc_position_embeddings]
                if self.mlm_config.get("sample_first_subword", True):
                    if len(word_start_idxes) > self.model.config.max_enc_position_embeddings:
                        sampled_idxes = sorted(
                            np.random.choice(
                                word_start_idxes,
                                self.model.config.max_enc_position_embeddings,
                                replace=False,
                            ).tolist()
                        )
                    elif len(word_start_idxes) < self.model.config.max_enc_position_embeddings:
                        not_word_start_idxes = list(set(total_subword_idxes).difference(set(word_start_idxes)))
                        sampled_idxes = np.random.choice(
                            not_word_start_idxes,
                            self.model.config.max_enc_position_embeddings - len(word_start_idxes),
                            replace=False,
                        ).tolist()
                        sampled_idxes = sorted(word_start_idxes + sampled_idxes)
                    else:
                        sampled_idxes = word_start_idxes

                enc_input_ids = enc_input_ids[sampled_idxes]
                enc_input_x_pos_ids = enc_input_x_pos_ids[sampled_idxes]
                enc_input_y_pos_ids = enc_input_y_pos_ids[sampled_idxes]
                enc_input_type_ids = enc_input_type_ids[sampled_idxes]

                filter_cl_word_candidates = set(sampled_idxes)
                filtered_cl_word_candidates = []
                for cand in cl_word_candidates:
                    if cand[0] in filter_cl_word_candidates:
                        filtered_cl_word_candidates.append((sampled_idxes.index(cand[0]), cand[1]))
                cl_word_candidates = filtered_cl_word_candidates

            if len(cl_word_candidates) > 0:
                cl_word_candidates_tensor[: len(cl_word_candidates)] = torch.tensor(cl_word_candidates)

                t_max = self.model.config.max_enc_position_embeddings - 1
                vis_max = int(img_scale_info["num_feasible_rows"] * img_scale_info["num_feasible_cols"]) - 1
                cl_word_candidates_tensor[:, 0] = torch.clamp(cl_word_candidates_tensor[:, 0], max=t_max)
                cl_word_candidates_tensor[:, 1] = torch.clamp(cl_word_candidates_tensor[:, 1], max=vis_max)

            assert (
                enc_input_ids.size(0)
                == enc_input_x_pos_ids.size(0)
                == enc_input_y_pos_ids.size(0)
                == enc_input_type_ids.size(0)
                == enc_attention_mask.size(0)
            )

            return (
                enc_input_ids,
                enc_input_x_pos_ids,
                enc_input_y_pos_ids,
                enc_input_type_ids,
                enc_attention_mask,
                cl_word_candidates_tensor,
            )
        except Exception as e:
            print("Exception in prepare_text_input:", e)
            import os
            import sys

            exc_type, exc_obj, exc_tb = sys.exc_info()
            fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
            print(exc_type, fname, exc_tb.tb_lineno)
            raise Exception()

    @staticmethod
    def get_random_question_generation_prompt(answer):
        return np.random.choice(
            [
                f"Given the image, generate a question whose answer is: {answer}. Question:",
                f"Based on the image, provide a question with the answer: {answer}. Question:",
                f'Given the visual representation, create a question for which the answer is "{answer}".',
                f"From the image provided, craft a question that leads to the reply: {answer}. Question:",
                f"Considering the picture, come up with a question where the answer is: {answer}.",
                f"Taking the image into account, generate an question that has the answer: {answer}. Question:",
            ]
        )

    @staticmethod
    def get_random_question_answering_prompt(query, config={}):
        if config.get("split", "train") != "train":
            return query

        if not config.get("llm_integration_enabled", False):
            return np.random.choice(
                [
                    query,
                    f"Question: {query}",
                    f"Given the image, answer the following question. {query}",
                    f'What is the answer to the following question? "{query}"',
                ],
                p=[0.55, 0.15, 0.15, 0.15],
            )

        return np.random.choice(
            [
                query,
                f"Q: {query}",
                f"Question: {query}",
                f"Given the image, answer the following question. {query}",
                f"Based on the image, respond to this question with a short answer: {query}. Answer:",
                f"Use the provided image to answer the question: {query} Provide your answer as short as possible:",
                f'What is the answer to the following question? "{query}"',
                f'The question "{query}" can be answered using the image. A short answer is',
            ]
        )

    @staticmethod
    def get_random_captioning_prompt():
        return np.random.choice(
            [
                "Explain the image.",
                "Use a few words to illustrate what is happening in the picture.",
                "Using language, provide a short account of the image.",
                "Please provide a short depiction of the picture.",
                "Could you use a few words to describe what you perceive in the photo?",
                "Can you briefly explain what you see in the image?",
                "Briefly describe the content of the image.",
                "Provide a description of what is presented in the photo.",
                "Write a description for the photo.",
                "Write a short description for the image.",
            ]
        )

    @staticmethod
    def get_random_text_reading_prompt():
        return np.random.choice(
            [
                "Read all texts",
                "Read all texts in the image.",
                "Read all characters in the image.",
                "Given the image, read all texts.",
                "Given the image, read all characters.",
            ]
        )

    @staticmethod
    def get_random_masked_text_prediction_prompt():
        return np.random.choice(
            [
                "Read masked texts.",
                "Read masked texts in the image.",
                "Given the image, read masked texts.",
                "Read all hidden texts that is covered by the mask area.",
            ]
        )

    def get_cream_decoder_sequences(self, query, answer):
        dec_input_ids = self.model.tokenizer(
            self.model.tokenizer.bos_token
            + query
            + self.model.tokenizer.sep_token
            + answer
            + self.model.tokenizer.eos_token,
            max_length=self.model.config.max_position_embeddings + 1,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
            add_special_tokens=False,
        )["input_ids"].squeeze(0)
        dec_label_ids = dec_input_ids[1:].clone()
        dec_input_ids = dec_input_ids[:-1]
        q_end_ids = (dec_input_ids == self.model.tokenizer.sep_token_id).nonzero().flatten()
        if len(q_end_ids):
            prompt_end_idx = q_end_ids[0]
            dec_label_ids[:prompt_end_idx] = self.ignore_id
        else:
            prompt_end_idx = torch.tensor(len(dec_label_ids) - 2).long()
        dec_label_ids[dec_label_ids == self.model.tokenizer.pad_token_id] = self.ignore_id
        return dec_input_ids, dec_label_ids, prompt_end_idx

    def prepare_qa(self, img, meta, qa_config):
        try:
            qa_pairs = meta["qa_pairs"]
            question_id = "N/A"
            if self.split == "train":
                qa_pair = qa_pairs[np.random.randint(len(qa_pairs))]
            else:
                qa_pair = qa_pairs[0]
                try:
                    if "question_id" in meta:
                        question_id = meta["question_id"]
                    else:
                        question_id = qa_pair[2]
                except:
                    question_id = "N/A"
            query = str(qa_pair[0]).replace("  ", " ").strip()

            if self.fix_seed_for_test:
                answer = str([x[1].replace("  ", " ").strip() for x in meta["qa_pairs"]])
            else:
                answer = str(qa_pair[1]).replace("  ", " ").strip()

            if query == "":
                raise Exception("query is none")

            if answer == "" and self.split == "train":
                raise Exception("answer is none")

            if np.random.rand() < qa_config.get("qg_p", 0.0) and qa_config["task"] != "cap":
                temp = self.get_random_question_generation_prompt(answer)
                answer = query
                query = temp
            else:
                if qa_config["task"] != "cap":
                    query = self.get_random_question_answering_prompt(query, config=qa_config)

            (
                img,
                img_attention_mask,
                scale_info,
            ) = self.model.image_encoder.prepare_input(
                img,
                max_patches=self.model.config.max_patches,
                verbose=False,
            )

            x_scale = scale_info["resized_width"] / scale_info["image_width"]
            y_scale = scale_info["resized_height"] / scale_info["image_height"]

            if self.model.config.encoder_layers > 0:
                (
                    enc_input_ids,
                    enc_input_x_pos_ids,
                    enc_input_y_pos_ids,
                    enc_input_type_ids,
                    enc_attention_mask,
                    cl_word_candidates_tensor,
                ) = self.prepare_text_input(
                    scale_info,
                    x_scale,
                    y_scale,
                    meta,
                    max_patches=self.model.config.max_patches,
                    verbose=False,
                    sample_words_for_cl=qa_config.get("max_cl_num_words", 5),
                )
            else:
                enc_input_ids = enc_input_x_pos_ids = enc_input_y_pos_ids = enc_input_type_ids = enc_attention_mask = cl_word_candidates_tensor = torch.tensor([])

            if self.llm_tokenizer is None:
                dec_input_ids, dec_label_ids, prompt_end_idx = self.get_cream_decoder_sequences(query, answer)
            else:
                llm_qa_length = self.llm_max_length - self.llm_vision_query_length
                if self.model.config.decoder_layers > 0:
                    dec_input_ids = self.model.tokenizer(
                        query,
                        max_length=self.llm_cream_query_length,
                        padding="max_length",
                        truncation=True,
                        return_tensors="pt",
                        add_special_tokens=False,
                    )["input_ids"].squeeze(0)
                else:
                    dec_input_ids = torch.tensor([])
                if self.split == "train":
                    prompt_end_idx = self.llm_tokenizer(
                        f"{query} {answer} {self.llm_tokenizer.eos_token}",
                        max_length=llm_qa_length,
                        padding="max_length",
                        truncation=True,
                        return_tensors="pt",
                        add_special_tokens=False,
                    )["input_ids"].squeeze(0)
                    prompt_end_idx_temp = self.llm_tokenizer(
                        f"{query} {self.llm_tokenizer.eos_token}",
                        max_length=llm_qa_length,
                        padding="max_length",
                        truncation=True,
                        return_tensors="pt",
                        add_special_tokens=False,
                    )["input_ids"].squeeze(0)
                    q_end_ids = (prompt_end_idx_temp == self.llm_tokenizer.eos_token_id).nonzero().flatten()
                    if not len(q_end_ids):
                        query_end_id = llm_qa_length - 5
                    else:
                        query_end_id = q_end_ids[0] - 1

                    dec_label_ids = prompt_end_idx.clone()

                    dec_label_ids[:query_end_id] = self.ignore_id
                    label_padding_ids = (dec_label_ids == self.llm_tokenizer.pad_token_id).nonzero().flatten()
                    if len(label_padding_ids):
                        dec_label_ids[label_padding_ids[0] + 1 :] = self.ignore_id
                    dec_label_ids = torch.cat(
                        [
                            torch.tensor([self.ignore_id] * self.llm_vision_query_length).long(),
                            dec_label_ids,
                        ],
                        dim=0,
                    )

                else:
                    prompt_end_idx = self.llm_tokenizer(
                        f"{query} {self.llm_tokenizer.eos_token}",
                        max_length=llm_qa_length,
                        padding="max_length",
                        truncation=True,
                        return_tensors="pt",
                        add_special_tokens=False,
                    )["input_ids"].squeeze(0)
                    q_end_ids = (prompt_end_idx == self.llm_tokenizer.eos_token_id).nonzero().flatten()
                    if not len(q_end_ids):
                        q_end_ids = torch.tensor([llm_qa_length - 5]).long()
                    dec_label_ids = q_end_ids[0]

            return (
                enc_input_ids,
                enc_input_x_pos_ids,
                enc_input_y_pos_ids,
                enc_input_type_ids,
                enc_attention_mask,
                img,
                img_attention_mask,
                dec_input_ids,
                dec_label_ids,
                query,
                answer,
                prompt_end_idx,
                question_id,
                cl_word_candidates_tensor,
            )
        except Exception as e:
            print("Exception in prepare_qa:", e)
            import os
            import sys

            exc_type, exc_obj, exc_tb = sys.exc_info()
            fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
            print(exc_type, fname, exc_tb.tb_lineno)
            raise Exception()

    def prepare_mlm(
        self,
        img,
        meta,
        mlm_config=dict(),
        has_line: bool = False,
        has_paragraph: bool = False,
        masking_indexes: List[int] = None,
        is_textread: bool = False,
        query=None,
    ):
        is_masked_text_visible = False if np.random.rand() < mlm_config.get("image_masking_p", 0.5) else True
        p = (
            mlm_config.get("level_read", [0.3, 0.7, 0.0, 0.0])
            if is_masked_text_visible
            else mlm_config.get("level_masked", [0.7, 0.3, 0.0, 0.0])
        )
        if has_line and p[2] == 0.0:
            p[2] += 0.1
        if has_paragraph and p[3] == 0.0:
            p[3] += 0.1
        if np.sum(p) != 1.0:
            p = np.array(p) / np.sum(p)
        level = np.random.choice(["word", "phrase", "line", "paragraph"], p=p)

        min_n, max_n = mlm_config.get(level, {}).get("range", [2, 5])
        if min_n < 1:
            min_n = 0
        if not is_masked_text_visible:
            if level == "word":
                max_n = mlm_config.get("max_mask_num_words_for_wordlevel", 10)
            else:
                max_n = mlm_config.get("max_mask_num_words", 4)

            if int(len(meta["words"]) * 0.4) < 1:
                raise Exception("[warning] no words in meta.")
            max_n = min(max_n, int(len(meta["words"]) * 0.4))

        sampled_word_ids = set()
        if masking_indexes is not None:
            sampled_word_ids = set([meta["words"][idx]["id"] for idx in masking_indexes])
        else:
            if level == "word":
                n = np.random.randint(min_n, max_n)
                sampled_word_idxes = rand_word_fn(
                    words=meta["words"],
                    n=n,
                )
                sampled_word_ids = set([meta["words"][idx]["id"] for idx in sampled_word_idxes])
            elif level == "phrase":
                if mlm_config.get("read_from_top_left", False):
                    start_idx = 0
                    n = len(meta["words"])
                    min_n = max_n = len(meta["words"])
                else:
                    n = np.random.randint(min_n, max_n)
                    start_idx = np.random.randint(len(meta["words"]))
                end_idx = min(start_idx + n, len(meta["words"]))
                sampled_word_ids = set([meta["words"][idx]["id"] for idx in range(start_idx, end_idx)])
            elif level == "line":
                if len(meta.get("lines", [])) > 0:
                    line = np.random.choice(meta["lines"])
                    sampled_word_ids = set(line["ids"])
                else:
                    return self.prepare_mlm(
                        img, meta, mlm_config, has_line, has_paragraph, is_textread=is_textread, query=query
                    )
            elif level == "paragraph":
                if len(meta.get("paragraphs", [])) > 0:
                    paragraph = np.random.choice(meta["paragraphs"])
                    sampled_word_ids = set(paragraph["ids"])
                else:
                    return self.prepare_mlm(
                        img, meta, mlm_config, has_line, has_paragraph, is_textread=is_textread, query=query
                    )
            else:
                raise Exception("wrong level selected")

        if (len(sampled_word_ids) < min_n) or len(sampled_word_ids) > max_n:
            return self.prepare_mlm(
                img, meta, mlm_config, has_line, has_paragraph, is_textread=is_textread, query=query
            )

        output = list()
        masked_ids = set()
        drw = ImageDraw.Draw(img, "RGBA")
        for word in meta["words"]:
            if word["id"] not in sampled_word_ids:
                continue

            output.append(word["text"])
            masked_ids.add(word["id"])
            bbox = word["bbox"] if "bbox" in word else word["boundingBox"]
            if not mlm_config.get("read_from_top_left", False):
                drw.polygon(
                    xy=[tuple(coord) for coord in bbox],
                    fill=(0, 128, 0, 128 if is_masked_text_visible else 255),
                )

        answer = " ".join(output).strip()
        if not answer:
            raise Exception("[warning] output text is empty.")

        (
            img,
            img_attention_mask,
            scale_info,
        ) = self.model.image_encoder.prepare_input(img, max_patches=self.model.config.max_patches, verbose=False)

        x_scale = scale_info["resized_width"] / scale_info["image_width"]
        y_scale = scale_info["resized_height"] / scale_info["image_height"]

        if query is None:
            if is_textread:
                query = self.get_random_text_reading_prompt()
            else:
                query = self.get_random_masked_text_prediction_prompt()

        if self.model.config.encoder_layers > 0:
            (
                enc_input_ids,
                enc_input_x_pos_ids,
                enc_input_y_pos_ids,
                enc_input_type_ids,
                enc_attention_mask,
                cl_word_candidates_tensor,
            ) = self.prepare_text_input(
                scale_info,
                x_scale,
                y_scale,
                meta,
                is_masked_text_visible=is_masked_text_visible,
                masked_ids=masked_ids,
                max_patches=self.model.config.max_patches,
                verbose=False,
                sample_words_for_cl=mlm_config.get("max_cl_num_words", 5),
            )
        else:
            enc_input_ids = enc_input_x_pos_ids = enc_input_y_pos_ids = enc_input_type_ids = enc_attention_mask = cl_word_candidates_tensor = torch.tensor([])

        dec_input_ids, dec_label_ids, prompt_end_idx = self.get_cream_decoder_sequences(query, answer)

        return (
            enc_input_ids,
            enc_input_x_pos_ids,
            enc_input_y_pos_ids,
            enc_input_type_ids,
            enc_attention_mask,
            img,
            img_attention_mask,
            dec_input_ids,
            dec_label_ids,
            answer,
            prompt_end_idx,
            level,
            is_masked_text_visible,
            cl_word_candidates_tensor,
        )

    @staticmethod
    def _get_image_and_meta_data_default(dataset, idx):
        image = Image.open(dataset[idx]["img_path"])
        meta_data = dataset[idx]["meta"]
        return image, meta_data

    @staticmethod
    def _get_image_and_meta_data_lmdb(dataset, idx):
        meta_data = dataset["meta_dataset_txn"].get(f"{idx}".encode("utf-8"))
        if meta_data is not None:
            meta_data = pickle.loads(meta_data)
        image = dataset["img_dataset_txn"].get(meta_data["image_index"].encode("utf-8"))
        if image is not None:
            image = Image.open(BytesIO(image)).convert("RGB")
        return image, meta_data

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        try:
            (
                dataset_idx,
                dataset,
                task,
                fixed_idx,
            ) = self.get_dataset_and_idx(idx)

            try:
                image, meta_data = self.get_image_and_meta_data(dataset, fixed_idx)
            except:
                image, meta_data = None, None

            if image is None or meta_data is None:
                return self.__getitem__(np.random.randint(self.dataset_length))

            if not meta_data.get("words", None):
                meta_data["words"] = []

            is_textread = False
            if task == "cap":
                assert meta_data.get("captions", None) is not None
                meta_data["qa_pairs"] = [
                    [self.get_random_captioning_prompt(), caption]
                    for caption in meta_data["captions"]
                    if caption.strip()
                ]
                if len(meta_data["qa_pairs"]) < 1:
                    raise Exception("[Warning] There is no caption in this meta data.")
            if "qa" in task or task == "cap":
                assert meta_data.get("qa_pairs", None) is not None
                qa_config = copy.deepcopy(self.qa_config)
                qa_config["task"] = task
                qa_config["split"] = self.split
                (
                    enc_input_ids,
                    enc_input_x_pos_ids,
                    enc_input_y_pos_ids,
                    enc_input_type_ids,
                    enc_attention_mask,
                    img,
                    img_attention_mask,
                    dec_input_ids,
                    dec_label_ids,
                    query,
                    answer,
                    prompt_end_idx,
                    question_id,
                    cl_word_candidates_tensor,
                ) = self.prepare_qa(image, meta_data, qa_config)
                level = "QA"
                is_masked_text_visible = "None"
            elif "mlm" in task:
                num_of_chars = 0
                filtered_words = []
                for word in meta_data["words"]:
                    if "text" in word and word["text"].strip():
                        num_of_chars += len(word["text"].strip())
                        filtered_words.append(word)
                if num_of_chars < 50 or num_of_chars > 5000:
                    return self.__getitem__(np.random.randint(self.dataset_length))
                if len(filtered_words) < 50 or len(filtered_words) > 1000:
                    return self.__getitem__(np.random.randint(self.dataset_length))
                meta_data["words"] = filtered_words

                if np.random.rand() < self.textread_batch_ratio and self.split == "train":
                    mlm_config = copy.deepcopy(self.textread_config)
                    is_textread = True
                else:
                    mlm_config = copy.deepcopy(self.mlm_config)

                (
                    enc_input_ids,
                    enc_input_x_pos_ids,
                    enc_input_y_pos_ids,
                    enc_input_type_ids,
                    enc_attention_mask,
                    img,
                    img_attention_mask,
                    dec_input_ids,
                    dec_label_ids,
                    answer,
                    prompt_end_idx,
                    level,
                    is_masked_text_visible,
                    cl_word_candidates_tensor,
                ) = self.prepare_mlm(
                    image,
                    meta_data,
                    mlm_config,
                    has_line=len(meta_data.get("lines", [])) > 0,
                    has_paragraph=len(meta_data.get("paragraph", [])) > 0,
                    is_textread=is_textread,
                )
                question_id = "None"
            else:
                raise Exception("[Not Implemented] Set `task` as `qa`, `cap`, or `mlm`.")

            if self.split == "train":
                return (
                    enc_input_ids,
                    enc_input_x_pos_ids,
                    enc_input_y_pos_ids,
                    enc_input_type_ids,
                    enc_attention_mask,
                    img,
                    img_attention_mask,
                    dec_input_ids,
                    dec_label_ids,
                    cl_word_candidates_tensor,
                    prompt_end_idx,
                    is_textread,
                )
            else:
                return (
                    enc_input_ids,
                    enc_input_x_pos_ids,
                    enc_input_y_pos_ids,
                    enc_input_type_ids,
                    enc_attention_mask,
                    img,
                    img_attention_mask,
                    dec_input_ids,
                    query,
                    dec_label_ids,
                    answer,
                    dataset_idx,
                    level,
                    is_masked_text_visible,
                    prompt_end_idx,
                    question_id,
                )

        except Exception as e:
            print("Exception in dataset:", e)
            exc_type, exc_obj, exc_tb = sys.exc_info()
            fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
            print(exc_type, fname, exc_tb.tb_lineno)
            return self.__getitem__(np.random.randint(self.dataset_length))
