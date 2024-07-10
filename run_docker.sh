mkdir hydra_outputs
sudo docker build -t mv2mp .
docker run -v $HI4D_DIR:/home/mv2mp/data -v ./hydra_outputs/:/home/mv2mp/outputs --gpus all mv2mp