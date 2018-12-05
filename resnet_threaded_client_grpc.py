# Copyright 2016 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Send JPEG image to tensorflow_model_server loaded with ResNet model.
   Now with more Threading!
"""

from __future__ import print_function

import grpc
import requests
import tensorflow as tf

from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis import prediction_service_pb2

import threading
import time

import ruamel.yaml as yaml
import random

import os

lock = threading.Lock()

# The image URL is the location of the image we should send to the server
IMAGE_URL = 'https://tensorflow.org/images/blogs/serving/cat.jpg'

tf.app.flags.DEFINE_string('server', 'localhost:8500',
                           'PredictionService host:port')
tf.app.flags.DEFINE_string('image', '', 'path to image in JPEG format')
FLAGS = tf.app.flags.FLAGS

#f = open('temp.txt', 'w')

def thread_request(sample,data,sf):
  channel = grpc.insecure_channel(FLAGS.server)
  stub = prediction_service_pb2.PredictionServiceStub(channel)
  # Send request
  # See prediction_service.proto for gRPC request/response details.
  request = predict_pb2.PredictRequest()
  request.model_spec.name = 'resnet'
  request.model_spec.signature_name = 'serving_default'
  request.inputs['image_bytes'].CopyFrom(
      tf.contrib.util.make_tensor_proto(data, shape=[1]))
  start = time.time()
  try:
    result = stub.Predict(request, 10.0)  # 10 secs timeout
    end = time.time()
    lock.acquire()
    sf.write('start: {}\n'.format(str(start))+'end: {}\ntotal time: {}\nSucess\n'.format(str(end),str(end-start)))
    lock.release()
  except:
    end = time.time()
    lock.acquire()
    sf.write('start: {}\n'.format(str(start))+'end: {}\ntotal time: {}\nFailure\n'.format(str(end),str(end-start)))
    lock.release()

def main(_):
  if FLAGS.image:
    with open(FLAGS.image, 'rb') as f:
      data = f.read()
  else:
    # Download the image since we weren't given one
    dl_request = requests.get(IMAGE_URL, stream=True)
    dl_request.raise_for_status()
    data = dl_request.content

  replicas = [1,10,20,30,40,50,60,70,80,90,100] 

  # Sample 20 random configurations
  for i in range(20):

    # Read in old docker-compose file
    with open('./docker/docker-compose-resnet.yml') as yf:
      list_doc = yaml.load(yf, Loader=yaml.RoundTripLoader)

    # Randomize configuration values
    list_doc['services']['web']['deploy']['replicas']=random.choice(replicas)
    list_doc['services']['web']['deploy']['resources']['limits']['cpus'] = "{}".format(round(random.uniform(0.1,1),2))
    list_doc['services']['web']['deploy']['resources']['limits']['memory'] = '{}M'.format(random.randint(100,1000))
    
    sf = open('./resnet_samples/sample{}'.format(i), 'w')
    yaml.dump(list_doc, sf, Dumper=yaml.RoundTripDumper)
    print(yaml.dump(list_doc, Dumper=yaml.RoundTripDumper))

    # Write new docker-compose file
    with open('./docker/docker-compose-resnet.yml', 'w') as yf:
      yaml.dump(list_doc, yf, Dumper=yaml.RoundTripDumper)

    # Remove old configuration
    os.system('docker stack rm rand')
    # This is because of an issue with docker return before the stack is removed
    os.system('sleep 10')
    os.system('docker stop $(docker ps -aq)')
    os.system('sleep 10')
    os.system('docker rm $(docker ps -aq)')
    os.system('sleep 10')
    
    # Start up new configuration
    os.system('docker stack deploy -c ./docker/docker-compose-resnet.yml rand')

    # Perform 100 threaded request for the configuration
    threads = []
    for j in range(100):
      t = threading.Thread(target = thread_request, args = (i,data,sf))
      threads.append(t)
      t.start()

    for t in threads:
      t.join()
    
    sf.close()

if __name__ == '__main__':
  tf.app.run()

'''
while threading.activeCount() > 1:
  pass
else:
  f.close()  
'''
