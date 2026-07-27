# Usage:
#    docker build --build-arg CACHE_BUST=$(date +%s) -f Dockerfile -t slowquant-docker .
#    docker run --rm slowquant-docker:latest

FROM python:3.10-slim

# Upgrade pip and install build tools
RUN pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir build pytest
# Force minimum versions
RUN pip install --no-cache-dir \
    networkx==3.0 \
    numba==0.60 \
    numpy==2.0 \
    pyscf==2.4 \
    scipy==1.13 \
    sympy==1.9 \
    qiskit==2.1.1 \
    qiskit-aer==0.17.2 \
    qiskit-ibm-provider==0.11.0 \
    qiskit-ibm-runtime==0.41.1 \
    qiskit-nature==0.7.2

# Do not cache from here
ARG CACHE_BUST=0
RUN echo "$CACHE_BUST"

WORKDIR /test

# Clone SlowQuant
RUN apt-get update && \
    apt-get install -y --no-install-recommends git && \
    git clone --depth 1 --branch pip https://github.com/erikkjellgren/SlowQuant.git /test/SlowQuant && \
    apt-get purge -y git && \
    apt-get autoremove -y && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /test/SlowQuant

# Build the package
RUN python -m build

# Install the generated wheel
RUN pip install dist/*.whl

# Run tests
CMD ["pytest"]
