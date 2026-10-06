"""Metrics from simulator-only semantic render annotations."""
import numpy as np


def class_pixel_fraction(data, id_to_labels, class_name):
    """Return the image fraction assigned to one semantic class.

    Replicator segmentation IDs are renderer ground truth. This value is for
    reward shaping/evaluation and must not be appended to policy observations.
    """
    labels = np.asarray(data)
    if labels.ndim == 3 and labels.shape[-1] == 1:
        labels = labels[..., 0]
    if labels.ndim != 2 or labels.size == 0 or not np.issubdtype(labels.dtype, np.integer):
        raise ValueError("Semantic segmentation must be a non-empty 2D integer image")
    if not isinstance(id_to_labels, dict):
        raise ValueError("Replicator idToLabels metadata is required")
    matching_ids = []
    for identity, metadata in id_to_labels.items():
        if not isinstance(metadata, dict):
            continue
        if str(metadata.get("class", "")).casefold() == str(class_name).casefold():
            try:
                matching_ids.append(int(identity))
            except (TypeError, ValueError):
                raise ValueError("Semantic class ID is not an integer") from None
    if not matching_ids:
        raise ValueError(f"Semantic class {class_name!r} is absent from the render")
    return float(np.isin(labels, matching_ids).mean())
