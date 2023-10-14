export https_proxy=http://127.0.0.1:65530
export http_proxy=http://127.0.0.1:65530
export all_proxy=socks5://127.0.0.1:65530
docker_user=tothemoon

# docker tag transformers:ds $docker_user/transformers:ds_$(date +%Y%m%d)
# docker push $docker_user/transformers:ds_$(date +%Y%m%d)
docker tag belle:$(date +%Y%m%d) $docker_user/belle:$(date +%Y%m%d)
docker push $docker_user/belle:$(date +%Y%m%d)