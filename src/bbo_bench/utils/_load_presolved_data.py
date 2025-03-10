import json

import boto3
import numpy as np
import pooch


def load_presolved_data(bucket_or_url, folder_name, black_box):
    """Load presolved data from either an S3 bucket or URL.

    Args:
        bucket_or_url (str): The S3 bucket name or URL to download the data package from.
        folder_name (str): The desired folder to access data from.
        black_box: black box object from poli-core being optimized

    Returns:
        presolver_x: The initial solutions to pass to the optimizer.
        presolver_y: The initial scores to pass to the optimizer.
    """
    if "s3" in bucket_or_url:
        ehrlich_data, presolved_data = _load_s3_data(
            bucket_or_url,
            folder_name,
        )
    else:
        ehrlich_data, presolved_data = _load_url_data(
            bucket_or_url,
            folder_name,
        )

    # Sanity check that black box is the same between this experiment and the data package
    print("Data package black box info:", ehrlich_data)

    # Reformat solutions from data package to match black box API
    presolver_x = [record["particle"] for record in presolved_data]
    vocab = black_box.alphabet

    def convert_to_alphabet(x):
        new_x = []
        for elem in x:
            new_x.append([vocab[int(e)] for e in elem])
        return new_x

    presolver_x = [eval(x) for x in presolver_x]
    presolver_x = np.array(convert_to_alphabet(presolver_x))
    presolver_y = (
        -1
        * np.array(
            [(record["score"]) for record in presolved_data], dtype=float
        )[:, None]
    )

    return presolver_x, presolver_y


def _load_s3_data(bucket, key_prefix):
    """Load data from S3 bucket.

    Args:
        bucket (str): The S3 bucket name.
        key_prefix (str): The prefix for the key of the data file.

    Returns:
        ehrlich_data: The Ehrlich function specification.
        presolved_data: The initial training data to pass to the optimizer.
    """
    s3_obj = boto3.client("s3")
    s3_clientobj_ehrlich = s3_obj.get_object(
        Bucket=bucket, Key=key_prefix + "ehrlich.jsonl"
    )
    s3_ehrlich_data = (
        s3_clientobj_ehrlich["Body"].read().decode("utf-8").splitlines()
    )

    s3_clientobj_presolved = s3_obj.get_object(
        Bucket=bucket, Key=key_prefix + "plain_pairs.jsonl"
    )
    s3_presolved_data = (
        s3_clientobj_presolved["Body"].read().decode("utf-8").splitlines()
    )

    ehrlich_data = [json.loads(line) for line in s3_ehrlich_data]
    presolved_data = [json.loads(line) for line in s3_presolved_data]

    return ehrlich_data, presolved_data


def _load_url_data(url, data_folder_name):
    """Uses pooch to fetch and load data from a URL.

    Args:
        url (str): The URL to download the data from.
        data_folder_name (str): The desired folder to access data from.

    Returns:
        ehrlich_data: The Ehrlich function specification.
        presolved_data: The initial training data to pass to the optimizer.
    """
    unpack = pooch.Untar(
        members=[
            data_folder_name + "/ehrlich.jsonl",
            data_folder_name + "/plain_pairs.jsonl",
        ]
    )

    fnames = pooch.retrieve(url=url, known_hash=None, processor=unpack)

    for fname in fnames:
        if fname.endswith("ehrlich.jsonl"):
            ehrlich_file_path = fname
        elif fname.endswith("plain_pairs.jsonl"):
            presolved_file_path = fname

    # Load the data
    with open(ehrlich_file_path) as f:
        ehrlich_data = [json.loads(line) for line in f]
    with open(presolved_file_path) as f:
        presolved_data = [json.loads(line) for line in f]

    return ehrlich_data, presolved_data
