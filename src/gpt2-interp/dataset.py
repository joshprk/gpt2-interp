import torch as t
from torch import Tensor
from transformer_lens import HookedTransformer

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

    Args:
        model (HookedTransformer): The model to tokenize the name pairs for

    Returns:
        Tensor: A tokenized name pairs tensor
    """
    answers = make_name_pairs()
    tokens = t.concat([model.to_tokens(list(names), prepend_bos=False).T for names in answers])
    return tokens
