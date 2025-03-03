import tempfile


def add_vocab_to_lambo_cfg(cfg, vocab: list):
    """Update given LaMBO2 config file with desired vocab

    Args:
        cfg: The LaMBO2 config file.
        vocab: The vocabulary to add to the config file.

    Returns:
        cfg: The updated LaMBO2 config file.
    """
    # Use black_box vocab to create temporary vocab file for LaMBO2
    vocab.extend(
        [
            "<cls>",
            "<pad>",
            "<eos>",
            "<unk>",
            ".",
            "-",
            "<mask>",
        ]
    )

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".txt", delete=False
    ) as temp_file:
        # Write each token to the file, one per line
        for token in vocab:
            temp_file.write(f"{token}\n")
        # Get the name of the temporary file
        temp_file_name = temp_file.name

    print(f"Tokens have been written to temporary file: {temp_file_name}")
    cfg.optimizer.tasks.protein_generation.protein_seq.tokenizer.vocab_file = (
        temp_file_name
    )
    cfg.optimizer.roots.protein_seq.tokenizer_transform.tokenizer.vocab_file = temp_file_name

    return cfg
