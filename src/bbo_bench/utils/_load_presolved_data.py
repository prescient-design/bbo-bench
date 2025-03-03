import json

import boto3
import numpy as np


def load_presolved_data(cfg, black_box, source="s3"):
    if source == "s3":
        s3_ehrlich_data, s3_presolved_data = load_s3_data(
            cfg.presolved_data_package.bucket,
            cfg.presolved_data_package.key_prefix,
        )
    ehrlich_data = [json.loads(line) for line in s3_ehrlich_data]

    presolved_data = [json.loads(line) for line in s3_presolved_data]

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


def load_s3_data(bucket, key_prefix):
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
    return s3_ehrlich_data, s3_presolved_data
