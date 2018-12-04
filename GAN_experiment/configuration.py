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


class Configuration:

    def __init__(self, compose_file=None):
        self.compose_file = compose_file
        self.max_replicas = 25
        self.compose_params = compose_params(None, None, None)
        self.load_compose_params()


    def set_compose_file(self, compose_file):
        self.compose_file = compose_file

    def load_compose_params(self):
        with open(self.compose_file, 'r') as stream:
            try:
                yaml_params = yaml.load(stream)
                print(yaml_params)
                replicas = yaml_params['services']['web']['deploy']['replicas']
                max_cpu = yaml_params['services']['web']['deploy']['resources']['limits']['cpus']
                max_memory = yaml_params['services']['web']['deploy']['resources']['limits']['memory']
                self.compose_params.replicas = replicas
                self.compose_params.max_cpu = str(max_cpu)
                self.compose_params.max_memory = re.sub("[^0-9]", "", max_memory)

            except yaml.YAMLError as exc:
                print(exc)


    def write_compose_params(self, replicas, max_cpu, max_mem):
        if replicas < 1:
            replicas = 1
        elif replicas > self.max_replicas:
            replicas = self.max_replicas

        if max_cpu < 0.2:
            max_cpu = 0.2
        elif max_cpu > 0.85:
            max_cpu = 0.85

        if max_mem < 500:
            max_mem = 500
        elif max_mem > 1000:
            max_mem = 1000

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


        yaml_params['services']['web']['deploy']['replicas'] = self.compose_params.replicas
        yaml_params['services']['web']['deploy']['resources']['limits']['cpus'] = self.compose_params.max_cpu
        yaml_params['services']['web']['deploy']['resources']['limits']['memory'] = self.compose_params.max_memory+'M'


        with open(self.compose_file, 'w') as outfile:
            yaml.dump(yaml_params, outfile, default_flow_style=False)


    def write_config(self, replicas, max_cpu, max_memory):
        self.write_compose_params(replicas, max_cpu, max_memory)

    def is_service_running(self, service_name):
        cmd="docker service ls --filter label=com.docker.stack.namespace="+service_name+" -q"
        time.sleep(2)
        pro = subprocess.Popen("exec "+cmd, stdout=subprocess.PIPE,shell=True)
        stdout, stderr = pro.communicate()
        output=stdout.decode("utf-8")
        lines=output.split("\n")
        pro.kill()
        print(lines)
        print("[is_service_running] "+str(len(lines)))
        return 1!=len(lines) # 1 for empty string

    def reload(self, replicas):
        service_name="gan_door_number"
        def waitForRemoval(docker_entity, service_name):
            cmd="docker "+docker_entity+" ls --filter label=com.docker.stack.namespace="+service_name+" -q"
            while (True):
                time.sleep(2)
                pro = subprocess.Popen("exec "+cmd, stdout=subprocess.PIPE,shell=True)
                stdout, stderr = pro.communicate()
                output=stdout.decode("utf-8")
                pro.kill()
                lines=output.split("\n")
                #errinfo=stderr.decode("utf-8")
                #errlines=errinfo.split("\n")
                print("[waitForRemoval] "+docker_entity)
                print("stdout:"+output)
                #print("stderr"+errlines)
                length = len(lines)
                if length == 1: # empty string
                    break

        def waitForAllContainersUp(service_name, replicas):
            cmd='docker container ls -q --format "table {{.Names}}"'
            
            while (True):
                num_of_containers_up=0
                time.sleep(2)
                pro = subprocess.Popen(cmd, stdout=subprocess.PIPE,shell=True)
                stdout, stderr = pro.communicate()
                pro.kill()
                output=stdout.decode("utf-8")
                lines=output.split("\n")
                #errinfo=stderr.decode("utf-8")
                #errlines=errinfo.split("\n")
                print("[waitForRemoval]")
                #print("stderr"+errlines)
                for line in lines:
                    print(line)
                    if service_name in line:
                        num_of_containers_up+=1
                print("[waitForAllContainersUp] {}/{} containers are up.".format(num_of_containers_up, replicas))
                if num_of_containers_up == replicas:
                    time.sleep(15)
                    break

        
        #Killing Previous App
        if self.is_service_running(service_name):
            status = subprocess.call(['docker', 'service', 'rm', service_name+"_web"])
            if status!=0:
                print("Error Killing Previous Service")
                return
            waitForRemoval("service", service_name)
            waitForRemoval("container", service_name)

        #Opening New Docker Service
        process = subprocess.call(['docker', 'stack', 'deploy', '-c', self.compose_file, service_name])
        if process != 0:
            print("Cannot Start New Docker Service")
            return
        waitForAllContainersUp(service_name, replicas)

        #time.sleep(warm_up_time_cost*replicas) # wait for the last up container to warm up for 30 seconds

    def get_config_as_array(self):
        self.load_compose_params()
        return np.array([self.compose_params.replicas, self.compose_params.max_cpu
                         , self.compose_params.max_memory])

    def create_random_config(self):
        replicas = random.randint(1, 25)
        max_cpu = random.random()
        max_memory = random.randint(500, 25000)

        # print(max_batch_size)
        return np.array([replicas, max_cpu, max_memory])

    def set_config_as_array(self, arr):
        self.write_config(arr[0], arr[1], arr[2])

    def print_configuration(self):
        self.load_compose_params()
        print("Replicas: %d" % self.compose_params.replicas)
        print("Max CPU: %s" % self.compose_params.max_cpu)
        print("Max Memory: %s" % self.compose_params.max_memory)


class SystemState:
    def __init__(self, inference_time=None, concurrency=None):
        # Default Configs
        compose_file = 'docker-compose.yml'
        self.config = Configuration(compose_file)
        self.inference_time = inference_time
        self.concurrency = concurrency

    def change_state(self, replicas, max_cpu, max_memory,
                        inference_time=None, concurrency=None):
        self.config.write_config(replicas, max_cpu, max_memory)
        self.inference_time = inference_time
        self.concurrency = concurrency

    def set_configuration(self, replicas, max_cpu, max_memory):
        self.config.write_config(replicas, max_cpu, max_memory)
        self.config.reload(replicas)

    def set_random_config(self):
        config_array = self.config.create_random_config()
        print("**************Setting Configuration:****************** ")
        # print(config_array)
        self.config.set_config_as_array(config_array)
        self.config.print_configuration()
        self.config.reload()

    def get_config_array(self):
        return self.config.get_config_as_array()


if __name__ == '__main__':

    # config = Configuration('docker-compose.yml', 1306)
    # config.print_configuration()
    # config.reload()max_batch_size, batch_timeout_micros, replicas, max_cpu, max_memory, gpu_freq

    state = SystemState()
    state.set_random_config()
    print(state.get_config_array())
    # state.config.print_configuration()





