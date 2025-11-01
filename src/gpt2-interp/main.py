import pandas as pd
import plotly.express as px
import torch as t
from rich import print as rprint
from rich.table import Column, Table
from torch import Tensor
from transformer_lens import ActivationCache, HookedTransformer, patching

MODEL_NAME = "gpt2-small"

PROMPT_TEMPLATES = [
    "While John and Mary went to the science fair,{} shared the project with",
    "When Tom and James were at the debate club,{} passed notes to",
    "During summer camp Dan and Sid were at,{} traded snacks with",
    "In the robotics workshop run by Martin and Amy,{} lent tools to",
]

NAME_PAIRS = [
    (" Mary", " John"),
    (" Tom", " James"),
    (" Dan", " Sid"),
    (" Martin", " Amy"),
]


def make_name_pairs() -> list[tuple[str, ...]]:
    return [names[::i] for names in NAME_PAIRS for i in (1, -1)]


def make_ioi_dataset(
    model: HookedTransformer,
    device: t.device | None = None,
    return_sentences: bool = False
) -> Tensor | tuple[Tensor, list[str]]:
    """
    Create an indirect object identification dataset from PROMPT_TEMPLATES and NAME_PAIRS.

    Args:
        model (HookedTransformer): The model to tokenize the dataset for
        device: (device | None): The device to move the token dataset to

    Returns:
        Tensor: The token dataset in memory
    """
    sentences = [
        prompt.format(name)
        for (prompt, names) in zip(PROMPT_TEMPLATES, NAME_PAIRS)
        for name in names[::-1]
    ]
    
    tokens = model.to_tokens(sentences)

    if device is not None:
        tokens = tokens.to(device)

    if return_sentences:
        return tokens, sentences
    else:
        return tokens


def tokenize_ioi_name_pairs(model: HookedTransformer) -> Tensor:
    """
    Return the tokenized array of correct/incorrect name pairs for the dataset created by
    `make_ioi_dataset`.
    """
    answers = make_name_pairs()
    tokens = t.concat([model.to_tokens(names, prepend_bos=False).T for names in answers])
    return tokens


def get_ioi_token_logits(logits: Tensor, answer: Tensor) -> Tensor:
    last_pos_logits = logits[:, -1, :]
    logits = last_pos_logits.gather(dim=-1, index=answer)
    return logits


def residual_stack_to_logit_diff(
    residual_stack: Tensor,
    cache: ActivationCache,
    logit_diff_dir: Tensor,
) -> Tensor:
    batch_size = residual_stack.size()[-2]
    scaled_residual_stack = cache.apply_ln_to_stack(residual_stack, layer=-1, pos_slice=-1)
    return t.einsum("...bd,bd->...", scaled_residual_stack, logit_diff_dir) / batch_size


def logit_ioi(
    model: HookedTransformer,
    sentences: list[str],
    logits: Tensor,
) -> Tensor:
    answers = tokenize_ioi_name_pairs(model)
    token_logits = get_ioi_token_logits(logits, answers)
    answer_pairs = make_name_pairs()

    table = Table(
        "Prompt",
        Column("Correct"),
        Column("Incorrect"),
        Column("Logit Difference"),
        title="Logit differences",
    )


    for prompt, answer, logits in zip(sentences, answer_pairs, token_logits):
        correct, incorrect = logits.unbind(dim=-1)
        logit_diff = correct - incorrect

        table.add_row(str(prompt), repr(answer[0]), repr(answer[1]), f"{logit_diff.item():.3f}")

    rprint(table)

    return token_logits


def ioi_metric(
    logits: Tensor,
    answer_tokens: Tensor,
    clean_logit_diff: float,
    corrupt_logit_diff: float,
) -> Tensor:
    """
    clean_corrupt_diff = 1.4
    corrupt_logit_diff = -1.0
    """
    clean = clean_logit_diff - corrupt_logit_diff
    patched = get_ioi_token_logits(logits, answer_tokens) - corrupt_logit_diff
    return (patched / clean).mean()


def layer_ioi_attribution(
    model: HookedTransformer,
    answers: Tensor,
    sentences: list[str],
    original_logit_diff: Tensor,
    cache: ActivationCache,
) -> None:
    # Logit attribution
    answers_residual_dir = model.tokens_to_residual_directions(answers)
    correct_residual_dir, incorrect_residual_dir = answers_residual_dir.unbind(dim=1)
    logit_diff_dir = correct_residual_dir - incorrect_residual_dir

    residual_stream = cache["resid_post", -1]
    token_residual_stream = residual_stream[:, -1, :]

    scaled_residual_stream = cache.apply_ln_to_stack(
        token_residual_stream,
        layer=-1,
        pos_slice=-1,
    )

    avg_logit_diff = t.einsum("bd,bd->", scaled_residual_stream, logit_diff_dir) / len(sentences)

    print(f"Average logit diff: {avg_logit_diff:.10f}")
    print(f"Logit diff: {original_logit_diff:.10f}")

    # Perform ablation on layers to find which ones perform IOI
    accumulated_residual, labels = cache.accumulated_resid(
        layer=-1,
        incl_mid=True,
        pos_slice=-1,
        return_labels=True,
    )

    logit_diff_per_layer = residual_stack_to_logit_diff(accumulated_residual, cache, logit_diff_dir)

    df = pd.DataFrame({
        "Layer": labels,
        "Logit Diff": logit_diff_per_layer.cpu().numpy(),
    })

    fig = px.line(
        df,
        title="Logit Difference From Accumulated Residual Stream",
        x="Layer",
        y="Logit Diff",
        labels={"Layer": "Layer", "Logit Diff": "Logit Diff"},
        width=800,
    )

    fig.show()


def head_attribution(cache: ActivationCache, logit_diff_dir: Tensor) -> None:
    head_resid, labels = cache.stack_head_results(layer=-1, pos_slice=-1, return_labels=True)
    logit_diffs = residual_stack_to_logit_diff(head_resid, cache, logit_diff_dir)

    fig = px.imshow(
        logit_diffs.cpu().reshape(cache.model.cfg.n_heads, cache.model.cfg.n_heads),
        labels={"x": "Head", "y": "Layer"},
        title="Logit difference from each head",
        width=600,
    )

    fig.show()


def activation_patching(model: HookedTransformer, tokens: Tensor, answer_tokens: Tensor) -> None:
    indices = [i + 1 if i % 2 == 0 else i - 1 for i in range(len(tokens))]
    clean_tokens = tokens
    corrupt_tokens = clean_tokens[indices]

    clean_logits, clean_cache = model.run_with_cache(clean_tokens)
    corrupt_logits, corrupt_cache = model.run_with_cache(corrupt_tokens)

    assert isinstance(clean_logits, Tensor), "clean_logits is a tensor"
    assert isinstance(corrupt_logits, Tensor), "corrupt_logits is a tensor"

    clean_logit_diff = get_ioi_token_logits(clean_logits, answer_tokens)
    corrupt, incorrect = clean_logit_diff.unbind(dim=-1)
    clean_logit_diff = (corrupt - incorrect).mean()

    corrupt_logit_diff = get_ioi_token_logits(corrupt_logits, answer_tokens)
    corrupt, incorrect = corrupt_logit_diff.unbind(dim=-1)
    corrupt_logit_diff = (corrupt - incorrect).mean()

    def patching_metric(logits: Tensor) -> Tensor:
        metric = ioi_metric(
            logits,
            answer_tokens,
            clean_logit_diff.item(),
            corrupt_logit_diff.item(),
        )

        return metric

    act_patch_resid_pre = patching.get_act_patch_resid_pre(
        model=model,
        corrupted_tokens=corrupt_tokens,
        clean_cache=clean_cache,
        patching_metric=patching_metric,
    )

    labels = [
        f"{tok} {i}"
        for i, tok in enumerate(model.to_str_tokens(clean_tokens[0]))
    ]

    fig = px.imshow(
        act_patch_resid_pre.cpu(),
        labels={"x": "Position", "y": "Layer"},
        x=labels,
        title="Activation patching",
        width=700,
    )

    fig.show()


def main() -> None:
    t.set_grad_enabled(False)

    device = t.device("cuda" if t.cuda.is_available() else "cpu")
    model = HookedTransformer.from_pretrained(
        MODEL_NAME,
        device=device,
        center_unembed=True,
        center_writing_weights=True,
        fold_ln=True,
        refactor_factored_attn_matrices=True,
    )

    tokens, sentences = make_ioi_dataset(model, device=device, return_sentences=True)
    logits, cache = model.run_with_cache(tokens)
    answers = tokenize_ioi_name_pairs(model)

    assert isinstance(logits, Tensor), "logits is a tensor output"
    assert isinstance(sentences, list), "sentences is a list of strings"

    logit_diff = logit_ioi(model, sentences, logits)

    correct, incorrect = logit_diff.unbind(dim=-1)
    original_logit_diff = (correct - incorrect).mean()

    layer_ioi_attribution(
        model,
        answers,
        sentences,
        original_logit_diff,
        cache,
    )

    answers_residual_dir = model.tokens_to_residual_directions(answers)
    correct_residual_dir, incorrect_residual_dir = answers_residual_dir.unbind(dim=1)
    logit_diff_dir = correct_residual_dir - incorrect_residual_dir

    head_attribution(cache, logit_diff_dir)

    activation_patching(model, tokens, answers)
