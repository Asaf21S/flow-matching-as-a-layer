import re

from src.fmlayer.data.specs import get_spec

WHITESPACE = re.compile(r"\s+")


def normalize_class_name(raw: str) -> str:
    """Normalise a raw torchvision class name for use in a CLIP prompt.

    Only underscores and repeated whitespace are touched, so FGVC-Aircraft variants such as
    ``F/A-18`` and ``DHC-8-100`` keep the tokens that carry their fine-grained signal.

    Args:
        raw: Class name exactly as torchvision provides it.

    Returns:
        The normalised class name.
    """
    return WHITESPACE.sub(" ", raw.replace("_", " ")).strip()


def normalize_class_names(raw_names: list[str]) -> list[str]:
    """Normalise a full class list.

    Args:
        raw_names: Class names in label-index order.

    Returns:
        The normalised names, in the same order.
    """
    return [normalize_class_name(name) for name in raw_names]


def build_prompts(dataset: str, raw_names: list[str]) -> list[str]:
    """Build the CLIP text prompt of every class, using the template fixed by the spec.

    Args:
        dataset: Dataset key, ``"dtd"`` or ``"aircraft"``.
        raw_names: Class names in label-index order.

    Returns:
        One prompt per class, in label-index order.
    """
    template = get_spec(dataset).prompt_template
    return [template.replace("{class}", normalize_class_name(name)) for name in raw_names]
