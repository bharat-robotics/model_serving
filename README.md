# model_serving

Load model for serving
docker run -p 8500:8500 \
--mount type=bind,source=$(pwd)/models/inception,target=/models/inception \
-e MODEL_NAME=inception -t tensorflow/serving &

Run the client script. Provide the server and path to images
python inception_client.py --server=127.0.0.1:8500 --image=./images
