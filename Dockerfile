# Select PyTorch image from nvidia
FROM nvcr.io/nvidia/pytorch:22.03-py3

ENV DEBIAN_FRONTEND noninteractive

WORKDIR /app/code

# Bootstrap pip
RUN wget https://bootstrap.pypa.io/get-pip.py && \
	python3 get-pip.py && \
	rm get-pip.py

# PdbPP: A better debuggin experience
RUN pip install pdbpp pygments
COPY ./.pdbrc.py /.pdbrc.py

# Install code requirements
COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt