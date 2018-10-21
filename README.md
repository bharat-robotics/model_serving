# model_serving

Export the model
python mnist_saved_model.py models/mnist

Load the model into a docker container for serving
docker run -p 8500:8500 \
--mount type=bind,source=$(pwd)/models/mnist,target=/models/mnist \
-e MODEL_NAME=mnist -t tensorflow/serving &

test the server
python mnist_client.py --num_tests=1000 --server=127.0.0.1:8500
