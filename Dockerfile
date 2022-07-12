FROM nvcr.io/nvidia/pytorch:21.06-py3

ENV TZ=Asia/Dubai
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

ENV PYTHONUNBUFFERED 1
ENV PYTHONDONTWRITEBYTECODE 1

RUN DEBIAN_FRONTEND=noninteractive apt-get -qq update \
 && DEBIAN_FRONTEND=noninteractive apt-get -qqy install screen libopenblas-base cmake python3-pip mc wget libportaudio2 ffmpeg git less nano libsm6 libxext6 libxrender-dev \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .

RUN python -m spacy download en_core_web_md
#RUN pip install llvmlite --ignore-installed
RUN pip3 install --upgrade pip
RUN pip3 install --no-deps -r requirements.txt
                          
ENTRYPOINT ["pwd"]