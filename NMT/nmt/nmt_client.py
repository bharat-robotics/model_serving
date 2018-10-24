# -*- coding : utf-8 -*-

from __future__ import print_function

import argparse
from nltk import word_tokenize
import time
import json
import os
import tensorflow as tf

from grpc.beta import implementations
from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis import prediction_service_pb2

PYTHONIOENCODING = "UTF-8"


def parse_translation_result(args, result):
    hypotheses = tf.make_ndarray(result.outputs["seq_output"])
    str1 = ' '.join(str(e) for e in hypotheses if e != "</s>")
    return str1


def translate(stub, model_name, tokens, timeout=5.0):
    request = predict_pb2.PredictRequest()
    request.model_spec.name = model_name
    request.model_spec.signature_name = 'predict_nmt'
    request.inputs["seq_input"].CopyFrom(tf.contrib.util.make_tensor_proto(tokens))
    print(request.inputs["seq_input"])
    xy = stub.Predict.future(request, timeout)
    return xy


def main():
    json_data = {}
    start = time.time()
    json_data['start'] = str(time.ctime(int(start)))

    parser = argparse.ArgumentParser(description="Translation client")
    parser.add_argument("--model_name", required=True,
                        help="model name (name of the file?)")
    parser.add_argument("--host", default="localhost",
                        help="model server host")
    parser.add_argument("--port", type=int, default=9000,
                        help="model server port")
    parser.add_argument("--timeout", type=float, default=10.0,
                        help="request timeout")
    parser.add_argument("--text", default="",
                        help="Untokenized input text")

    args = parser.parse_args()
    channel = implementations.insecure_channel(args.host, args.port)
    stub = prediction_service_pb2.beta_create_PredictionService_stub(channel)

    tokens_list = []
    if args.text != "":
        json_data['input'] = args.text
        text = args.text
        tokens = text.split()
        tokens_list.append(tokens)

    token_count = 0
    for tokens in tokens_list:
        trans = translate(stub, args.model_name, tokens, timeout=args.timeout)
        result = trans.result()
        best_result = parse_translation_result(args, result)
        json_data['result_' + str(token_count)] = best_result

    end = time.time()
    json_data['duration'] = str(round(end - start, 3)) + " sec"
    json_data['end'] = str(time.ctime(int(end)))
    json_result = json.dumps(json_data, sort_keys=True)
    print(json_result)


if __name__ == "__main__":
    main()
