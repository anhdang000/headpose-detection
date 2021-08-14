# Start FROM Nvidia PyTorch image https://ngc.nvidia.com/catalog/containers/nvidia:pytorch
FROM nvcr.io/nvidia/pytorch:21.05-py3

# Copy content
ADD . /workspace
WORKDIR /workspace

# Install python dependencies
COPY requirements.txt .
RUN python -m pip install --upgrade pip
RUN pip install -r requirements.txt
RUN pip install gunicorn
RUN chmod +x boot.sh

ENV FLASK_APP server.py
EXPOSE 80
ENTRYPOINT ["./boot.sh"]