from __future__ import print_function

import numpy as np
from recordtype import recordtype
import yaml
import utils
import subprocess
import random
import re
import time

compose_params = recordtype('compose_params', 'replicas  max_cpu max_memory')
batching_params = recordtype('batching_params', 'max_batch_size batch_timeout_micros')


class Configuration:

    def __init__(self, batching_file=None, compose_file=None, gpu_freq=None):
        self.batching_file = batching_file
        self.compose_file = compose_file
        self.allowed_gpu_freq = np.array([1306, 1293, 1280, 1267, 1254, 1241, 1228, 1215, 1202, 1189, 1176, 1163, 1150, 1137, 1124, 1110, 1097,
                                          1084, 1071, 1058, 1045, 1032, 1019, 1006, 993, 980, 967, 954, 941, 928, 915, 901, 888, 875, 862, 849,
                                          836, 823, 810, 797, 784, 771, 758, 745, 732, 719, 705, 692, 679, 666, 653, 640, 627, 614, 601, 588, 575,
                                          562, 549, 405, 392, 379, 366, 353, 340, 327, 324, 314, 301, 288, 275, 270, 267, 265, 263, 261, 259, 257,
                                          255, 253, 251, 249, 247, 245, 243, 241, 239, 237, 235, 233, 231, 229, 226, 224, 222, 220, 218, 216, 214,
                                          212, 210, 208, 206, 204, 202, 200, 198, 196, 194, 192, 190, 188, 186,
                                          183, 181, 179, 177, 175, 173, 171, 169, 167, 165, 163, 161, 159, 157, 155, 153, 151, 149, 147, 145, 143, 140, 138, 136])
        self.allowed_batch_size = np.array([4, 8, 16, 32, 64, 96, 128])
        self.min_batch_timeout = 100
        self.max_batch_timeout = 100000
        self.max_replicas = 20
        self.compose_params = compose_params(None, None, None)
        self.batching_params = batching_params(None, None)
        self.load_compose_params()
        self.load_batching_params()
        self.set_gpu_freq(gpu_freq)

    def set_batching_file(self, batching_file):
        self.batching_file = batching_file

    def set_compose_file(self, compose_file):
        self.compose_file = compose_file

    def load_compose_params(self):
        with open(self.compose_file, 'r') as stream:
            try:
                yaml_params = yaml.load(stream)
                replicas = yaml_params['services']['tf']['deploy']['replicas']
                max_cpu = yaml_params['services']['tf']['deploy']['resources']['limits']['cpus']
                max_memory = yaml_params['services']['tf']['deploy']['resources']['limits']['memory']
                self.compose_params.replicas = replicas
                self.compose_params.max_cpu = str(max_cpu)
                self.compose_params.max_memory = re.sub("[^0-9]", "", max_memory)

            except yaml.YAMLError as exc:
                print(exc)

    def load_batching_params(self):
        with open(self.batching_file, 'r') as files:
            lines = files.readlines()
            max_batch_size = lines[0].split(':')[1].split('}')[0]
            batch_timeout_micros = lines[1].split(':')[1].split('}')[0]
            self.batching_params.max_batch_size = int(max_batch_size)
            self.batching_params.batch_timeout_micros = int(batch_timeout_micros)

    def write_batching_params(self, max_batch_size, batch_timeout_micros):
        max_batch_size = utils.find_nearest_config(self.allowed_batch_size, max_batch_size)

        if batch_timeout_micros < self.min_batch_timeout:
            batch_timeout_micros = self.min_batch_timeout
        elif batch_timeout_micros > self.max_batch_timeout:
            batch_timeout_micros = self.max_batch_timeout

        self.batching_params.max_batch_size = int(max_batch_size)
        self.batching_params.batch_timeout_micros = int(batch_timeout_micros)
        self.dump_batching_params()

    def dump_batching_params(self):
        with open(self.batching_file, 'w+') as file:
            file.write('max_batch_size { value: %d }\n' % self.batching_params.max_batch_size)
            file.write('batch_timeout_micros { value: %d }\n' % self.batching_params.batch_timeout_micros)
            file.write('pad_variable_length_inputs: true\n')
            file.write('num_batch_threads {value: 8}')

    def write_compose_params(self, replicas, max_cpu, max_mem):
        if replicas < 1:
            replicas = 1
        elif replicas > self.max_replicas:
            replicas = self.max_replicas

        if max_cpu < 0.2:
            max_cpu = 0.2
        elif max_cpu > 0.5:
            max_cpu = 0.5

        if max_mem < 500:
            max_mem = 500
        elif max_mem > 10000:
            max_mem = 10000

        self.compose_params.max_cpu = str(round(max_cpu, 2))
        self.compose_params.max_memory = str(int(max_mem/100)*100)
        self.compose_params.replicas = int(replicas)
        self.dump_compose_params()

    def dump_compose_params(self):
        with open(self.compose_file, 'r') as stream:
            try:
                yaml_params = yaml.load(stream)
            except yaml.YAMLError as exc:
                print(exc)


        yaml_params['services']['tf']['deploy']['replicas'] = self.compose_params.replicas
        yaml_params['services']['tf']['deploy']['resources']['limits']['cpus'] = self.compose_params.max_cpu
        yaml_params['services']['tf']['deploy']['resources']['limits']['memory'] = self.compose_params.max_memory+'M'


        with open(self.compose_file, 'w') as outfile:
            yaml.dump(yaml_params, outfile, default_flow_style=False)

    def set_gpu_freq(self, gpu_freq):
        self.gpu_freq = utils.find_nearest_config(self.allowed_gpu_freq, gpu_freq)

    def write_config(self, max_batch_size, batch_timeout_micros, replicas, max_cpu, max_memory, gpu_freq):
        self.write_compose_params(replicas, max_cpu, max_memory)
        self.write_batching_params(max_batch_size, batch_timeout_micros)
        self.set_gpu_freq(gpu_freq)

    def reload(self):
        status = self.run_gpu_with_freq()
        if status != 0:
            print("Error Setting GPU Freq")
            return

        #Killing Previous App
        status = subprocess.call(['docker', 'stack', 'rm', 'opennmtapp'])
        if status!=0:
            print("Error Killing Previous Service")
            return

        time.sleep(30)

        #Opening New Docker Service
        process = subprocess.call(['docker', 'stack', 'deploy', '-c', self.compose_file, 'opennmtapp'])
        if process != 0:
            print("Cannot Start New Docker Service")
            return

    def run_gpu_with_freq(self):
        output = subprocess.call(['nvidia-smi', '--application-clocks=900,'+str(self.gpu_freq)])
        return output

    def get_config_as_array(self):
        self.load_batching_params()
        self.load_compose_params()
        return np.array([self.batching_params.max_batch_size, self.batching_params.batch_timeout_micros, self.compose_params.replicas, self.compose_params.max_cpu
                         , self.compose_params.max_memory, self.gpu_freq])

    def create_random_config(self):
        max_batch_size = random.randint(1, 130)
        batch_timeout_micros = random.randint(100, 100000)
        replicas = random.randint(1, 20)
        max_cpu = random.random()
        max_memory = random.randint(500, 10000)
        gpu_freq = random.randint(100, 1400)

        # print(max_batch_size)
        return np.array([max_batch_size, batch_timeout_micros, replicas, max_cpu, max_memory, gpu_freq])

    def set_config_as_array(self, arr):
        self.write_config(arr[0], arr[1], arr[2], arr[3], arr[4], arr[5])

    def print_configuration(self):
        self.load_batching_params()
        self.load_compose_params()
        print("Max Batch Size: %d" % self.batching_params.max_batch_size)
        print("Batch Timeout Micros: %d" % self.batching_params.batch_timeout_micros)
        print("Replicas: %d" % self.compose_params.replicas)
        print("Max CPU: %s" % self.compose_params.max_cpu)
        print("Max Memory: %s" % self.compose_params.max_memory)
        print("GPU Freq: %d" % self.gpu_freq)


class SystemState:
    def __init__(self, batch_size=None, inference_time=None, concurrency=None):
        # Default Configs
        batching_file = 'batching_parameters.txt'
        compose_file = 'docker-compose.yml'
        gpu_freq = 1306
        self.config = Configuration(batching_file, compose_file, gpu_freq)
        self.batch_size = batch_size
        self.inference_time = inference_time
        self.concurrency = concurrency

    def change_state(self, gpu_freq, max_batch_size, batch_timeout_micros, replicas, max_cpu, max_memory,
                        batch_size=None, inference_time=None, concurrency=None):
        self.config.write_config(gpu_freq, max_batch_size, batch_timeout_micros, replicas, max_cpu, max_memory)
        self.batch_size = batch_size
        self.inference_time = inference_time
        self.concurrency = concurrency

    def set_batch_size(self,batch_size):
        self.batch_size = batch_size

    def set_concurrency(self, con):
        self.concurrency = con

    def set_inference_time(self, time):
        self.inference_time = time

    def set_configuration(self, max_batch_size, batch_timeout_micros, replicas, max_cpu, max_memory, gpu_freq):
        self.config.write_config(gpu_freq, max_batch_size, batch_timeout_micros, replicas, max_cpu, max_memory)
        self.config.reload()

    def set_random_config(self):
        config_array = self.config.create_random_config()
        print("**************Setting Configuration:****************** ")
        # print(config_array)
        self.config.set_config_as_array(config_array)
        self.config.print_configuration()
        self.config.reload()

    def get_config_array(self):
        return self.config.get_config_as_array()

    def get_system_array(self):
        self.config.load_batching_params()
        self.config.load_compose_params()
        return np.array([self.config.batching_params.max_batch_size, self.config.batching_params.batch_timeout_micros, self.config.compose_params.replicas,
                         self.config.compose_params.max_cpu, self.config.compose_params.max_memory, self.config.gpu_freq, self.batch_size, self.inference_time,
                         self.concurrency])


if __name__ == '__main__':

    # config = Configuration('batching_parameters.txt', 'docker-compose.yml', 1306)
    # config.write_batching_params(100, 10)
    # config.write_compose_params(5, 0.6, 100)
    # config.set_gpu_freq(1200)
    # config.print_configuration()
    # config.reload()max_batch_size, batch_timeout_micros, replicas, max_cpu, max_memory, gpu_freq
    # Killing Previous App

    state = SystemState()
    state.set_random_config()
    print(state.get_config_array())
    # state.config.print_configuration()




