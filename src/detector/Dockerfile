# Dockerfile for the restorer
FROM stemo:llm

# copy python script to the image
COPY main.py /root/main.py

# set the working directory to /root
WORKDIR /root

# install scikit-learn using pip3
RUN pip3 install scikit-learn

# run python script when the container starts
CMD ["python3", "main.py"]