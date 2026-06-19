def validate_dataset_shapes(data, input_len=None, output_len=None,
                            input_dim=None, num_nodes=None,
                            target_dim=None):
    """Validate train/val/test arrays against CLI/model shape settings."""
    splits = ("train", "val", "test")
    first_x = data["x_train"]
    first_y = data["y_train"]

    if first_x.ndim != 4:
        raise ValueError(f"x_train must be 4D [S,T,N,C], got {first_x.shape}")
    if first_y.ndim != 4:
        raise ValueError(f"y_train must be 4D [S,H,N,C], got {first_y.shape}")

    actual = {
        "input_len": first_x.shape[1],
        "num_nodes": first_x.shape[2],
        "input_dim": first_x.shape[3],
        "output_len": first_y.shape[1],
        "target_dim": first_y.shape[3],
    }

    expected = {
        "input_len": input_len,
        "output_len": output_len,
        "input_dim": input_dim,
        "num_nodes": num_nodes,
        "target_dim": target_dim,
    }
    for name, value in expected.items():
        if value is not None and actual[name] != value:
            raise ValueError(
                f"Dataset {name} mismatch: expected {value}, got {actual[name]}. "
                "Regenerate data or pass matching CLI arguments."
            )

    for split in splits:
        x = data[f"x_{split}"]
        y = data[f"y_{split}"]
        if x.ndim != 4 or y.ndim != 4:
            raise ValueError(f"{split} arrays must be 4D, got x={x.shape}, y={y.shape}")
        if x.shape[1:] != first_x.shape[1:]:
            raise ValueError(
                f"x_{split} shape mismatch: expected trailing shape "
                f"{first_x.shape[1:]}, got {x.shape[1:]}"
            )
        if y.shape[1:] != first_y.shape[1:]:
            raise ValueError(
                f"y_{split} shape mismatch: expected trailing shape "
                f"{first_y.shape[1:]}, got {y.shape[1:]}"
            )
        if x.shape[2] != y.shape[2]:
            raise ValueError(
                f"{split} node count mismatch: x has {x.shape[2]}, y has {y.shape[2]}"
            )

    return actual
