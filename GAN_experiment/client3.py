#!/usr/bin/env python2.7

"""A client that talks to tensorflow_model_server loaded with mnist model.

The client downloads test images of mnist data set, queries the service with
such test images to get predictions, and calculates the inference error rate.

'''
Send JPEG image to tensorflow_model_server loaded with GAN model.

Hint: the code has been compiled together with TensorFlow serving
and not locally. The client is called in the TensorFlow Docker container
'''

Typical usage example:

    mnist_client.py --num_tests=100 --server=localhost:9000
"""

from __future__ import print_function

import sys
import threading

import numpy
from argparse import ArgumentParser

import os
#import signal
#import subprocess
from os import listdir
from os.path import isfile, join

from matplotlib import pyplot as plt
import time

# Communication to TensorFlow server via gRPC
import grpc
from grpc.beta import implementations
import tensorflow as tf

# TensorFlow serving stuff to send messages
from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis import prediction_service_pb2
from tensorflow.contrib.util import make_tensor_proto
from tensorflow_serving.apis import prediction_service_pb2_grpc

# docker configuration
import configuration

# for Bayesian optimization
from bayes_opt import BayesianOptimization

g_host="localhost"
g_port=8500
g_image_dir="/home/lluo/2T/jsu/mls-project/hello2"
g_concurrency=20
g_re_configure_docker=True

class _ResultCounter(object):
    """Counter for the prediction results."""
    def __init__(self, num_tests, concurrency):
        self._num_tests = num_tests
        self._concurrency = concurrency
        self._error = 0
        self._done = 0
        self._active = 0
        self._cnt_rt = 0
        self._response_times=numpy.zeros(num_tests)
        self._condition = threading.Condition()

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
            return round(self._error / float(self._num_tests), 4)

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

    def get_response_times(self):
        with self._condition:
            while self._done != self._num_tests:
                self._condition.wait()
            return self._response_times



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
        exception = result_future.exception()
        if exception:
            result_counter.inc_error()
            print(exception)
        else:
            '''
            sys.stdout.write('.')
            sys.stdout.flush()
            '''

            response = numpy.array(result_future.result().outputs['scores'].float_val)
            prediction = numpy.argmax(response)
#           if label != prediction:
#               result_counter.inc_error()

        result_counter.set_response_time(test_idx, time.time()-start_time)
        result_counter.inc_done()
        result_counter.dec_active()
    return _callback

def do_inference(host, port, images_dir, concurrency):
    """Tests PredictionService with concurrent requests.

    Args:
        host, port: Host:port address of the PredictionService.
        work_dir: The full path of working directory for test data set.
        concurrency: Maximum number of concurrent requests.

    Returns:
        The classification error rate.

    Raises:
        IOError: An error occurred processing test data set.
    """

    print("[do_inference] Concurrency: "+str(concurrency))
    image_paths = [join(images_dir,f) for f in listdir(images_dir) if isfile(join(images_dir, f))]
    num_tests=len(image_paths)
    result_counter = _ResultCounter(num_tests, concurrency)

    # Sending requests to servers
    for idx in range(0,num_tests):
        print("Testing image "+str(idx+1))
        start_time = time.time()
        channel = grpc.insecure_channel(host+":"+port)#implementations.insecure_channel(host, int(port))
        stub = prediction_service_pb2_grpc.PredictionServiceStub(channel)#prediction_service_pb2.beta_create_PredictionService_stub(channel)
        with open(image_paths[idx], "rb") as f:
            data = f.read()
            request = predict_pb2.PredictRequest()
            request.model_spec.name = 'gan'
            request.model_spec.signature_name = 'predict_images'
            request.inputs['images'].CopyFrom(make_tensor_proto(data, shape=[1]))
            result_counter.throttle()
            result_future = stub.Predict.future(request, 60.0)  # 5 seconds
            result_future.add_done_callback(_create_rpc_callback(result_counter, idx, start_time))
    print("Number of testing images : {}".format(num_tests))


    # Computer average response time and plot
    start=0
    end=num_tests # not included test
    num_of_outliers_to_exclude=num_tests-(end-start)
    response_times=(result_counter.get_response_times())[start:end]
    ave_time = sum(response_times)/len(response_times)
    print('\nAveraged Response Time: {}.'.format(ave_time))
    #plt.plot(range(1, len(response_times)+1), response_times, 'bo')
    #plt.xlabel("Test Index", fontsize=24)
    #plt.ylabel("Response Time (s)",fontsize=24)
    #plt.title("Concurrency: "+str(concurrency), fontsize=24)
    #plt.tick_params(labelsize=20)
    #plt.tight_layout()
    #plt.savefig("./response-time-numoftests-"+str(num_tests)+"-numofoutliers-"+str(num_of_outliers_to_exclude)+"-concurrency-"+str(concurrency)+".png")
    #plt.savefig("./response-time-numoftests-"+str(num_tests)+"-concurrency-"+str(concurrency)+".png")
    #plt.close()
    return ave_time, result_counter.get_error_rate()

def myplot(x, y, xlabel, ylabel, title, fig_path, init_points, best_bp_point_idx):
    best_idx=best_bp_point_idx-1
    plt.plot(x[:init_points], y[:init_points], 'ro', label='Initial Ramdon Configuration')
    plt.plot(x[init_points:], y[init_points:], 'bo', label="Bayesian Optimization Configuration")
    plt.plot(x[best_idx], y[best_idx], 'go', label="Best Configuration")
    plt.xlabel(xlabel, fontsize=24)
    plt.ylabel(ylabel, fontsize=24)
    plt.title(title, fontsize=24)
    plt.tick_params(labelsize=20)
    plt.tight_layout()
    plt.savefig(fig_path)
    plt.close()


def parse_args():
    parser = ArgumentParser(description='Request a TensorFlow server for a prediction on the image')
    parser.add_argument('-s', '--server',
                        dest='server',
                        default='172.17.0.2:8500',
                        help='prediction service host:port')
    parser.add_argument('-i', '--image_path',
                        dest='image_path',
                        default='',
                        help='path to images folder',)
    parser.add_argument('-c', '--concurrency',
                        dest='concurrency',
                        default=1,
                        help='number of concurrent requests')
	
    args = parser.parse_args()

    host, port = args.server.split(':')

    return host, port, args.image_path, args.concurrency

#TCP_USER_TIMEOUT=300000 python client.py -s localhost:8500 -i ./hello -c 1
def singleRun(
    num_containers,
    max_cpu,
    max_mem,
    host,
    port,
    image_dir,
    concurrency,
    re_configure_docker):
    if re_configure_docker:
        state=configuration.SystemState()
        # 1 container, 1 CPU and 1000M
        state.set_configuration(num_containers, max_cpu, max_mem)
        print(state.get_config_array())
    return do_inference(host, port, image_dir, concurrency)

# for BP
def target_func_BP(
    num_containers,
    max_cpu,
    max_mem):

    num_containers=int(round(num_containers))
    max_cpu=round(max_cpu, 2)
    max_mem=int(round(max_mem))
    print("[num_containers: {}, max_cpu: {}, max_mem: {}.]".format(num_containers, max_cpu, max_mem))
    if g_re_configure_docker:
        state=configuration.SystemState()
        state.set_configuration(num_containers, max_cpu, max_mem)
        print(state.get_config_array())
    ave_time, error_rate=do_inference(g_host, str(g_port), g_image_dir, g_concurrency)
    return -ave_time




def main():
    #host, port, image_dir, concurrency = parse_args()
    #if image_dir == "":
    #    print('Please specifiy testing image directory')
    #    return
    #if not host or not port:
    #    print('please specify server host:port')
    #    return

    #concurrency=int(concurrency)
#    if concurrency == 0:
#        print("Testing the impact of concurrency")
#        con_list=[1, 10, 20, 30, 40, 50, 60, 70, 80, 80, 90, 100]
#        #con_list=[1,2]
#        #num_errors_list=[]
#        error_rates_list=[]
#        ave_res_times=[]
#        for con in con_list:
#            print("Testing with concurrency "+str(con))
#            ave_res_time, error_rate = singleRun(1, 1, 1000, host, port, image_dir, con, True)
#            #num_errors_list.append(num_errors)
#            error_rates_list.append(error_rate)
#            ave_res_times.append(ave_res_time)
#        #myplot(con_list, error_rates_list, "Concurrency", "Ratio of timeouts in all requests", "1 Container: 1 CPU + 1000M", "timeouts.png")
#        myplot(con_list, ave_res_times, "Concurrency", "Average Response Time", "1 Container: 1 CPU + 1000M", "ave_response_time.png")
#        print("Average Response Time List: {}".format(ave_res_times))
#    else:
#        print("Regular Testing")
#        ave_res_time,  error_rate = singleRun(1,1,1000,host,port,image_dir,concurrency, True)
#        print("average response time: {}, rates of timeouts {}".format(ave_res_time, error_rate))

    init_points=10
    iterations=50
    #g_host=host
    #g_port=port
    #g_image_dir=image_dir
    #g_concurrency=concurrency 


    pbounds = {'num_containers': (1, 10), 'max_cpu': (0.3, 0.85), 'max_mem':(500, 1000)}

    bo = BayesianOptimization(
        f=target_func_BP,
        pbounds=pbounds,
        verbose=2,
        random_state=1)

    bo.maximize(init_points=2, n_iter=2, acq="ei", xi=1e-4)

    x=[]
    y=[]
    best_bp_point_idx=-1
    best_ave_time=sys.float_info.max
    best_rm_point_idx=-1
    best_rm_ave_time=sys.float_info.max


    for i, res in enumerate(bo.res):
        x.append(i)
        cur_target=round(-res['target'], 4)
        y.append(cur_target)
        if i>init_points:
            if best_ave_time > cur_target:
                best_ave_time=cur_target
                best_bp_point_idx=i
        else if i>=1 and i<=init_points
            if best_rm_ave_time > cur_target:
                best_rm_ave_time=cur_target
                best_rm_point_idx=i
        print("Iteration {}: \n\t{}".format(i, res))
    fig_path=str(g_concurrency)+" concurrency.png"
    result_path=str(g_concurrency)+" concurrency.text"
    title=str(init_points)+" Initial Points, "+str(iterations)+" BP Points"
    with open(result_path, "w") as output:
        output.write("Initial_Points: "+str(init_points)+"\n")
        output.write("BP_Points: "+str(iterations)+"\n")
        output.write("Best_BP_Point:"+str(best_bp_point_idx)+"\n")
        output.write("Best_average_response_time:"+str(best_ave_time)+"\n")
        output.write("Best_RM_Point:"+str(best_rm_point_idx)+"\n")
        output.write("Best_RM_average_response_time:"+str(best_rm_ave_time)+"\n")

        output.write("X:")
        for i in x:
            output.write(str(i)+",")
        output.write("\n")
        output.write("Y:")
        for i in y:
            output.write(str(i)+",")
        output.write("\n")
    myplot(x, y, "Iteration Index", "Average Response Time", title, fig_path, init_points, best_bp_point_idx)


if __name__ == '__main__':
    main()
