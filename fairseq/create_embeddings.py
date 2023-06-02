#!/usr/bin/env python3 -u
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""
Translate raw text with a trained model. Batches data on-the-fly.
"""

import ast
import fileinput
import logging
import math
import os
import pickle
import sys
import time
from argparse import Namespace
from collections import namedtuple, OrderedDict

import numpy as np
import torch

from fairseq import checkpoint_utils, distributed_utils, options, tasks, utils
from fairseq.data import RoundRobinZipDatasets
from fairseq.dataclass.configs import FairseqConfig
from fairseq.dataclass.utils import convert_namespace_to_omegaconf
from fairseq.token_generation_constraints import pack_constraints, unpack_constraints
from fairseq_cli.generate import get_symbols_to_strip_from_output

from language_triple_dataset import LanguageTripleDataset

logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=os.environ.get("LOGLEVEL", "INFO").upper(),
    stream=sys.stdout,
)
logger = logging.getLogger("fairseq_cli.interactive")


Batch = namedtuple("Batch", "ids src_tokens src_lengths urls constraints")
Translation = namedtuple("Translation", "src_str hypos pos_scores alignments")


def buffered_read(input, buffer_size):
    buffer = []
    with fileinput.input(files=[input], openhook=fileinput.hook_encoded("utf-8")) as h:
        for src_str in h:
            buffer.append(src_str.strip())
            if len(buffer) >= buffer_size:
                yield buffer
                buffer = []

    if len(buffer) > 0:
        yield buffer

def filter_indices_by_size(self, indices, max_positions=None):
    """
    Filter each sub-dataset independently, then update the round robin to work
    on the filtered sub-datasets.
    """

    def _deep_until_language_pair(dataset):
        if isinstance(dataset, LanguageTripleDataset):
            return dataset
        if hasattr(dataset, "tgt_dataset"):
            return _deep_until_language_pair(dataset.tgt_dataset)
        if hasattr(dataset, "dataset"):
            return _deep_until_language_pair(dataset.dataset)
        raise Exception(f"Don't know how to unwrap this dataset: {dataset}")

    if not isinstance(max_positions, dict):
        max_positions = {k: max_positions for k in self.datasets.keys()}
    ignored_some = False
    for key, dataset in self.datasets.items():
        dataset = _deep_until_language_pair(dataset)
        self._ordered_indices[key], ignored = dataset.filter_indices_by_size(
            self._ordered_indices[key], max_positions[key]
        )
        if len(ignored) > 0:
            ignored_some = True
            logger.warning(
                f"{len(ignored)} samples from {key} have invalid sizes and will be skipped, "
                f"max_positions={max_positions[key]}, first few sample ids={ignored[:10]}"
            )
    # Since we are modifying in place the _ordered_indices,
    # it's not possible anymore to return valid ignored indices.
    # Hopefully the extra debug information print above should be enough to debug.
    # Ideally we would receive ignore_invalid_inputs so that we could have
    # a proper error message.
    return (np.arange(len(self)), [0] if ignored_some else [])

def build_dataset_for_inference(task, src_tokens, src_lengths, urls, constraints=None):
        if constraints is not None:
            raise NotImplementedError(
                "Constrained decoding with the multilingual_translation task is not supported"
            )

        lang_pair = "%s-%s" % (task.args.source_lang, task.args.target_lang)
        round_robin_dataset = RoundRobinZipDatasets(
            OrderedDict(
                [
                    (
                        lang_pair,
                        task.alter_dataset_langtok(
                            LanguageTripleDataset(
                                src_tokens, src_lengths, task.source_dictionary, urls=urls
                            ),
                            src_eos=task.source_dictionary.eos(),
                            src_lang=task.args.source_lang,
                            tgt_eos=task.target_dictionary.eos(),
                            tgt_lang=task.args.target_lang,
                        ),
                    )
                ]
            ),
            eval_key=lang_pair,
        )

        RoundRobinZipDatasets.filter_indices_by_size = filter_indices_by_size

        return round_robin_dataset

def make_batches(lines, cfg, task, max_positions, encode_fn):
    def encode_fn_target(x):
        return encode_fn(x)

    urls = []
    for i, line in enumerate(lines):
        if ' | ' in line:
            url, lines[i] = line.split(' | ', 1)
            urls.append(url)
        else:
            urls.append('')

    if cfg.generation.constraints:
        # Strip (tab-delimited) contraints, if present, from input lines,
        # store them in batch_constraints
        batch_constraints = [list() for _ in lines]
        for i, line in enumerate(lines):
            if "\t" in line:
                lines[i], *batch_constraints[i] = line.split("\t")

        # Convert each List[str] to List[Tensor]
        for i, constraint_list in enumerate(batch_constraints):
            batch_constraints[i] = [
                task.target_dictionary.encode_line(
                    encode_fn_target(constraint),
                    append_eos=False,
                    add_if_not_exist=False,
                )
                for constraint in constraint_list
            ]

    if cfg.generation.constraints:
        constraints_tensor = pack_constraints(batch_constraints)
    else:
        constraints_tensor = None

    tokens, lengths = task.get_interactive_tokens_and_lengths(lines, encode_fn)

    itr = task.get_batch_iterator(
        dataset=build_dataset_for_inference(
            task, tokens, lengths, urls, constraints=constraints_tensor
        ),
        max_tokens=cfg.dataset.max_tokens,
        max_sentences=cfg.dataset.batch_size,
        max_positions=max_positions,
        ignore_invalid_inputs=cfg.dataset.skip_invalid_size_inputs_valid_test,
    ).next_epoch_itr(shuffle=False)
    for i, batch in enumerate(itr):
        ids = batch["id"]
        src_tokens = batch["net_input"]["src_tokens"]
        src_lengths = batch["net_input"]["src_lengths"]
        urls = batch["net_input"]["urls"]
        constraints = batch.get("constraints", None)

        yield Batch(
            ids=ids,
            src_tokens=src_tokens,
            src_lengths=src_lengths,
            urls=urls,
            constraints=constraints,
        )


def main(cfg: FairseqConfig):
    if isinstance(cfg, Namespace):
        cfg = convert_namespace_to_omegaconf(cfg)

    start_time = time.time()
    total_translate_time = 0

    utils.import_user_module(cfg.common)

    if cfg.interactive.buffer_size < 1:
        cfg.interactive.buffer_size = 1
    if cfg.dataset.max_tokens is None and cfg.dataset.batch_size is None:
        cfg.dataset.batch_size = 1

    assert (
        not cfg.generation.sampling or cfg.generation.nbest == cfg.generation.beam
    ), "--sampling requires --nbest to be equal to --beam"
    assert (
        not cfg.dataset.batch_size
        or cfg.dataset.batch_size <= cfg.interactive.buffer_size
    ), "--batch-size cannot be larger than --buffer-size"

    logger.info(cfg)

    # Fix seed for stochastic decoding
    if cfg.common.seed is not None and not cfg.generation.no_seed_provided:
        np.random.seed(cfg.common.seed)
        utils.set_torch_seed(cfg.common.seed)

    use_cuda = torch.cuda.is_available() and not cfg.common.cpu

    # Setup task, e.g., translation
    task = tasks.setup_task(cfg.task)

    # Load ensemble
    overrides = ast.literal_eval(cfg.common_eval.model_overrides)
    logger.info("loading model(s) from {}".format(cfg.common_eval.path))
    models, _model_args = checkpoint_utils.load_model_ensemble(
        utils.split_paths(cfg.common_eval.path),
        arg_overrides=overrides,
        task=task,
        suffix=cfg.checkpoint.checkpoint_suffix,
        strict=(cfg.checkpoint.checkpoint_shard_count == 1),
        num_shards=cfg.checkpoint.checkpoint_shard_count,
    )

    # Set dictionaries
    src_dict = task.source_dictionary
    tgt_dict = task.target_dictionary

    # Optimize ensemble for generation
    for model in models:
        if model is None:
            continue
        if cfg.common.fp16:
            model.half()
        if use_cuda and not cfg.distributed_training.pipeline_model_parallel:
            model.cuda()
        model.prepare_for_inference_(cfg)

    # Initialize generator
    generator = task.build_generator(models, cfg.generation)

    # Handle tokenization and BPE
    tokenizer = task.build_tokenizer(cfg.tokenizer)
    bpe = task.build_bpe(cfg.bpe)

    def encode_fn(x):
        if tokenizer is not None:
            x = tokenizer.encode(x)
        if bpe is not None:
            x = bpe.encode(x)
        return x

    def decode_fn(x):
        if bpe is not None:
            x = bpe.decode(x)
        if tokenizer is not None:
            x = tokenizer.decode(x)
        return x

    # Load alignment dictionary for unknown word replacement
    # (None if no unknown word replacement, empty if no path to align dictionary)
    align_dict = utils.load_align_dict(cfg.generation.replace_unk)

    max_positions = utils.resolve_max_positions(
        task.max_positions(), *[model.max_positions() for model in models]
    )

    if cfg.generation.constraints:
        logger.warning(
            "NOTE: Constrained decoding currently assumes a shared subword vocabulary."
        )

    if cfg.interactive.buffer_size > 1:
        logger.info("Sentence buffer size: %s", cfg.interactive.buffer_size)
    logger.info("NOTE: hypothesis and token scores are output in base 2")
    logger.info("Type the input sentence and press return:")
    start_id = 0
    embeddings = []
    for inputs in buffered_read(cfg.interactive.input, cfg.interactive.buffer_size):
        results = []
        for batch in make_batches(inputs, cfg, task, max_positions, encode_fn):
            bsz = batch.src_tokens.size(0)
            src_tokens = batch.src_tokens
            src_lengths = batch.src_lengths
            urls = batch.urls
            constraints = batch.constraints
            if use_cuda:
                src_tokens = src_tokens.cuda()
                src_lengths = src_lengths.cuda()
                if constraints is not None:
                    constraints = constraints.cuda()

            sample = {
                "net_input": {
                    "src_tokens": src_tokens,
                    "src_lengths": src_lengths,
                },
            }
            translate_start_time = time.time()
            embeddings.append(task.inference_step(
                generator, models, sample, constraints=constraints
            ) + urls)

        # update running id_ counter
        start_id += len(inputs)

    with open(os.environ['CODE_EMBEDDING_PATH'], 'wb') as f:
        pickle.dump(embeddings, f)

    logger.info(
        "Total time: {:.3f} seconds; translation time: {:.3f}".format(
            time.time() - start_time, total_translate_time
        )
    )


def cli_main():
    parser = options.get_interactive_generation_parser()
    args = options.parse_args_and_arch(parser)
    distributed_utils.call_main(convert_namespace_to_omegaconf(args), main)


if __name__ == "__main__":
    cli_main()
