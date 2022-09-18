#FROM python:3.9-buster
FROM tensorflow/tensorflow:latest-gpu
RUN apt update && apt install -y libgl1-mesa-glx
COPY ./ /app
WORKDIR /app
RUN pip3 install -r requirements.txt
RUN mkdir -p /root/.cache/torch/hub/checkpoints/
RUN curl https://download.pytorch.org/models/mobilenet_v3_small-047dcff4.pth -o /root/.cache/torch/hub/checkpoints/mobilenet_v3_small-047dcff4.pth

# get models begin
# RUN mkdir weights
# RUN cd weights
# RUN wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1O4zXptRuDUjTsY-apjcjsTpnfV-kzhJX' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1O4zXptRuDUjTsY-apjcjsTpnfV-kzhJX" -O weights/vit_huge.zip && rm -rf /tmp/cookies.txt &
# RUN unzip vit_huge.zip
# RUN cd ../
# get models end

CMD ["python3","run.py"]
