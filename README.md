# model_serving
Run the serving image as a daemon
docker run -d --name serving_base tensorflow/serving

Copy RestNet model data to the container's model folder 
docker cp ./models/resnet serving_base:/models/resnet

Commit the container to serving the ResNet model:
docker commit --change "ENV MODEL_NAME resnet" serving_base \
  $USER/resnet_serving

Stop the serving base container
docker kill serving_base
docker rm serving_base

Start the container with the ResNet model exposing the gRPC port 8500
docker run -p 8500:8500 -t $USER/resnet_serving &

Query the server with resnet_client_grpc.py. The client downloads an image and sends it over gRPC for classification into ImageNet categories
python tensorflow_serving/example/resnet_client_grpc.py
