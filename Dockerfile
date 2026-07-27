FROM python:3.10-slim

WORKDIR /test

# Install git and build tools
RUN apt-get update && \
    apt-get install -y git && \
    rm -rf /var/lib/apt/lists/*

# Clone SlowQuant
RUN git clone --depth 1 --branch pip https://github.com/erikkjellgren/SlowQuant.git

WORKDIR /test/SlowQuant

# Upgrade pip and install build tools
RUN pip install --upgrade pip
RUN pip install build pytest
# Force minimum versions
RUN pip install \
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

# Build the package
RUN python -m build

# Install the generated wheel
RUN pip install dist/*.whl

# Run tests
CMD ["pytest"]
