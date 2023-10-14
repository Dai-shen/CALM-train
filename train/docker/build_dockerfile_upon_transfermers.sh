export https_proxy=http://127.0.0.1:65530
export http_proxy=http://127.0.0.1:65530
export all_proxy=socks5://127.0.0.1:65530

wget https://raw.githubusercontent.com/huggingface/transformers/main/docker/transformers-pytorch-deepspeed-latest-gpu/Dockerfile -O transformers.dockerfile
docker build --network host --build-arg HTTP_PROXY=$http_proxy -t transformers:ds -f transformers.dockerfile .
docker build --network host --build-arg HTTP_PROXY=$http_proxy -t belle:$(date +%Y%m%d) -f belle.dockerfile .
