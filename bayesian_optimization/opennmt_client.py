#!/usr/bin/env python

"""A client that talks to tensorflow_model_server loaded with opennmt model.

The client uses the text file to generate test data for conversion, set, queries the service with
such test images to get predictions, and calculates the inference error rate.

Typical usage example:

    mnist_client.py --num_tests=100 --server=localhost:9000
"""

import sys
import threading
import argparse
import pyonmttok

import grpc
import numpy
import tensorflow as tf

from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis import prediction_service_pb2_grpc

from matplotlib import pyplot as plt
import time
import random
from nltk import sent_tokenize

from configuration import Configuration, SystemState

concurrencies = [10, 25, 50, 100, 150, 200]
possible_batch_sizes = [2, 4, 8, 16, 32, 64]
#
# concurrencies = [2]
# possible_batch_sizes = [2]

class _ResultCounter(object):

  def __init__(self, num_tests, concurrency, batch_size=2):
    self._num_tests = num_tests
    self._concurrency = concurrency
    self._batch_size = batch_size
    self._error = 0
    self._done = 0
    self._active = 0
    self._cnt_rt = 0
    self._response_times = numpy.zeros(num_tests)
    self._condition = threading.Condition()
    self._start_time = 0
    self._end_time = 0

  def inc_error(self):
    with self._condition:
      self._error += 1

  def inc_done(self):
    with self._condition:
      self._done += 1
      self._condition.notify()

  def dec_active(self):
    with self._condition:
      self._active -= 1
      self._condition.notify()

  def get_error_rate(self):
    with self._condition:
      while self._done != self._num_tests:
        self._condition.wait()
      return self._error / float(self._num_tests)


  def get_error(self):
    with self._condition:
      while self._done != self._num_tests:
        self._condition.wait()
      return self._error

  def throttle(self):
    with self._condition:
      while self._active == self._concurrency:
        self._condition.wait()
      self._active += 1

  def inc_cnt_rt(self):
      with self._condition:
          self._cnt_rt +=1
          self._condition.notify()

  def set_response_time(self, test_idx, response_time):
      with self._condition:
          if test_idx > 0 and test_idx <= self._num_tests:
              self._response_times[test_idx-1] = response_time

  def set_start_time(self, indx, time):
      with self._condition:
          if indx == 0:
              self._start_time = time

  def set_end_time(self, indx, time):
      with self._condition:
        if indx == self._num_tests-1:
            self._end_time = time

  def get_response_times(self):
      with self._condition:
        while self._done != self._num_tests:
          self._condition.wait()
        return self._response_times

  def get_times(self):
      with self._condition:
        while self._done != self._num_tests:
          self._condition.wait()
        return self._start_time, self._end_time

def _create_rpc_callback(result_counter, test_idx, start_time):
  """Creates RPC callback function.

  Args:
    label: The correct label for the predicted example.
    result_counter: Counter for the prediction result.
  Returns:
    The callback function.
  """
  def _callback(result_future):
    """Callback function.

    Calculates the statistics for the prediction result.

    Args:
      result_future: Result future of the RPC.
    """
    result_counter.set_response_time(test_idx, time.time()-start_time)
    result_counter.set_end_time(test_idx, time.time())

    exception = result_future.exception()
    if exception:
      result_counter.inc_error()
      print(exception)
    else:
      sys.stdout.flush()

    result_counter.inc_done()
    result_counter.dec_active()
  return _callback


def pad_batch(batch_tokens):
  """Pads a batch of tokens."""
  lengths = [len(tokens) for tokens in batch_tokens]
  max_length = max(lengths)
  for tokens, length in zip(batch_tokens, lengths):
    if max_length > length:
      tokens += [""] * (max_length - length)
  return batch_tokens, lengths, max_length


def extract_prediction(result):
  """Parses a translation result.

  Args:
    result: A `PredictResponse` proto.

  Returns:
    A generator over the hypotheses.
  """
  batch_lengths = tf.make_ndarray(result.outputs["length"])
  batch_predictions = tf.make_ndarray(result.outputs["tokens"])
  for hypotheses, lengths in zip(batch_predictions, batch_lengths):
    # Only consider the first hypothesis (the best one).
    best_hypothesis = hypotheses[0]
    best_length = lengths[0] - 1  # Ignore </s>
    yield best_hypothesis[:best_length]

def create_tokens(input_file):
  with open(input_file, 'r') as file:
    data = file.read()
  return sent_tokenize(data)


def write_state(state, start_time, end_time, error):
    log_file = open('docker-logs.txt', 'a')
    configs = state.get_system_array()
    log_file.write('%d,%d,%d,%f,%d,%d,%d,%f,%d,%f,%f,%d\n' %
                   (int(configs[0]), int(configs[1]), int(configs[2]), float(configs[3]), int(configs[4]), int(configs[5]),
                    int(configs[6]), float(configs[7]), int(configs[8]), float(start_time), float(end_time), int(error)))

    log_file.close()

def do_inference(host, port, sentencepiece_model, model_name, timeout):

  channel = grpc.insecure_channel("%s:%d" % (host, port))
  stub = prediction_service_pb2_grpc.PredictionServiceStub(channel)
  tokenizer = pyonmttok.Tokenizer("none", sp_model_path=sentencepiece_model)
  secure_random = random.SystemRandom()

  all_tokens = create_tokens('english_data.txt')

  state = SystemState()

  for conc in concurrencies:
    for batch_size in possible_batch_sizes:
        for conf in range(10):
            state.set_random_config()
            time.sleep(20)
            state.set_concurrency(conc)
            state.set_batch_size(batch_size)
            result_counter = _ResultCounter(conc, conc, batch_size)
            for i in range(conc):
                batch_input = secure_random.sample(all_tokens, batch_size)
                batch_values = [tokenizer.tokenize(text)[0] for text in batch_input]
                batch_tokens, lengths, max_length = pad_batch(batch_values)

                request = predict_pb2.PredictRequest()
                request.model_spec.name = model_name
                request.inputs["tokens"].CopyFrom(
                  tf.make_tensor_proto(batch_tokens, shape=(batch_size, max_length)))
                request.inputs["length"].CopyFrom(
                  tf.make_tensor_proto(lengths, shape=(batch_size,)))

                result_counter.throttle()
                start_time = time.time()
                result_counter.set_start_time(i, start_time)
                future = stub.Predict.future(request, timeout)
                future.add_done_callback(
                _create_rpc_callback(result_counter, i, start_time))
                # result = future.result()

                # batch_output = [tokenizer.detokenize(prediction) for prediction in extract_prediction(result)]

                # for input_text, output_text in zip(batch_input, batch_output):
                #   print("{} ||| {}".format(input_text, output_text))
            inference_time = sum(result_counter.get_response_times())/batch_size/conc
            st_time, end_time = result_counter.get_times()
            state.set_inference_time(inference_time)
            write_state(state, st_time, end_time, result_counter.get_error())


def main():
  parser = argparse.ArgumentParser(description="Translation client example")
  parser.add_argument("--model_name", default='ende',
                      help="model name")
  parser.add_argument("--sentencepiece_model", default='ende/1539080952/assets.extra/wmtende.model',
                      help="path to the sentence model")
  parser.add_argument("--host", default="localhost",
                      help="model server host")
  parser.add_argument("--port", type=int, default=8500,
                      help="model server port")
  parser.add_argument("--timeout", type=float, default=100000,
                      help="request timeout")
  args = parser.parse_args()

  do_inference(args.host, args.port, args.sentencepiece_model, args.model_name, args.timeout)


if __name__ == "__main__":
  main()