FROM nvidia/cuda:11.5.0-devel-ubuntu20.04
MAINTAINER musk

# udpate timezone
RUN apt-get update \
    &&  DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends tzdata

RUN TZ=Asia/Taipei \
    && ln -snf /usr/share/zoneinfo/$TZ /etc/localtime \
    && echo $TZ > /etc/timezone \
    && dpkg-reconfigure -f noninteractive tzdata 

RUN apt-get update
RUN apt-get install -y curl python3-distutils vim \
	git make unzip wget openssh-server software-properties-common

# install python
RUN apt-get -y install python3.9
RUN echo "alias python=python3.9" >> ~/.bashrc
#RUN source ~/.bashrc

# install pip
RUN curl -O https://bootstrap.pypa.io/get-pip.py
RUN /usr/bin/python3.9 get-pip.py

# Download tp annealing project
WORKDIR /home/
RUN git clone https://github.com/jerrycci/QuantumAnnealing.git
RUN pip install -r /home/QuantumAnnealing/requirement.txt

# ssh setup
RUN mkdir -p /var/run/sshd
RUN sed -i '/^#/!s/PermitRootLogin .*/PermitRootLogin yes/' /etc/ssh/sshd_config
RUN service ssh restart
RUN echo 'root:root' | chpasswd
RUN echo "Port 22" >> /etc/ssh/sshd_config
RUN echo "PasswordAuthentication yes" >> /etc/ssh/sshd_config

EXPOSE 22

CMD /etc/init.d/ssh/restart
