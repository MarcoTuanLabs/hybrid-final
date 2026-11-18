"""Greedy MLX inference with explicit cache boundaries and measured forward positions."""
from __future__ import annotations

import copy
import time
import mlx.core as mx
import numpy as np
from mlx_lm import load
from mlx_lm.models.cache import make_prompt_cache


class Engine:
    def __init__(self, path):
        self.model, self.tokenizer = load(str(path))
        self.language = self.model.language_model
        self.positions = 0

    def encode(self, text):
        return self.tokenizer.encode(text, add_special_tokens=False)

    def chat(self, text):
        return self.tokenizer.apply_chat_template(
            [dict(role="user", content=text)], tokenize=False,
            add_generation_prompt=True, enable_thinking=False,
        )

    def forward(self, ids, cache):
        hidden = self.model.model(mx.array([ids]), cache=cache)
        self.positions += len(ids)
        last = hidden[:, -1:]
        if self.language.args.tie_word_embeddings:
            logits = self.language.model.embed_tokens.as_linear(last)
        else:
            logits = self.language.lm_head(last)
        return logits[0, 0].astype(mx.float32)

    def prefill(self, tokens):
        cache = make_prompt_cache(self.model)
        hidden = self.model.model(mx.array([tokens]), cache=cache)
        self.positions += len(tokens)
        mx.eval(hidden)
        return cache

    def answer(self, tokens, cache=None, stop=24):
        if cache is None:
            cache = make_prompt_cache(self.model)
        output = []
        first = None
        for _ in range(stop):
            logits = self.forward(tokens, cache)
            if first is None:
                first = np.asarray(logits)
            token = int(mx.argmax(logits).item())
            if token in self.tokenizer.eos_token_ids:
                break
            output.append(token)
            if "\n" in self.tokenizer.decode(output):
                break
            tokens = [token]
        mx.synchronize()
        return self.tokenizer.decode(output).split('\n', 1)[0].strip(), first

    def factual(self, question):
        prompt = 'Answer with only the short factual answer.\n' + question
        return self.answer(self.encode(self.chat(prompt)))[0].strip('*` ').rstrip('.')
