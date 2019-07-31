# MLModelScope MXNet Agent

[![Build Status](https://travis-ci.org/rai-project/mxnet.svg?branch=master)](https://travis-ci.org/rai-project/mxnet)
[![Build Status](https://dev.azure.com/dakkak/rai/_apis/build/status/mxnet)](https://dev.azure.com/dakkak/rai/_build/latest?definitionId=17)
[![Go Report Card](https://goreportcard.com/badge/github.com/rai-project/mxnet)](https://goreportcard.com/report/github.com/rai-project/mxnet)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

[![](https://images.microbadger.com/badges/version/carml/mxnet:ppc64le-gpu-latest.svg)](https://microbadger.com/images/carml/mxnet:ppc64le-gpu-latest> 'Get your own version badge on microbadger.com') [![](https://images.microbadger.com/badges/version/carml/mxnet:ppc64le-cpu-latest.svg)](https://microbadger.com/images/carml/mxnet:ppc64le-cpu-latest 'Get your own version badge on microbadger.com') [![](https://images.microbadger.com/badges/version/carml/mxnet:amd64-cpu-latest.svg)](https://microbadger.com/images/carml/mxnet:amd64-cpu-latest 'Get your own version badge on microbadger.com') [![](https://images.microbadger.com/badges/version/carml/mxnet:amd64-gpu-latest.svg)](https://microbadger.com/images/carml/mxnet:amd64-gpu-latest 'Get your own version badge on microbadger.com')

This is the MXNet agent for [MLModelScope](mlmodelscope.org), an open-source framework and hardware agnostic, extensible and customizable platform for evaluating and profiling ML models across datasets / frameworks / systems, and within AI application pipelines.

Currently it has most of the models from Gluon Model Zoo built in. More built-in models are comming.
One can evaluate the **~80** models on any systems of insterest with either local MXNet installation or MXNet docker images.

Check out [MLModelScope](mlmodelscope.org) and welcome to contribute.

## Installation

Install go if you have not done so. Please follow [Go Installation](https://docs.mlmodelscope.org/installation/source/golang).

Download and install the MLModelScope MXNet Agent:

```
go get -v github.com/rai-project/mxnet

```

The agent requires The MXNet C library and other Go packages.

### Go packages

You can install the dependency through `go get`.

```
cd $GOPATH/src/github.com/rai-project/mxnet
go get -u -v ./...
```

Or use [Dep](https://github.com/golang/dep).

```
dep ensure -v
```

This installs the dependency in `vendor/`.

Note: The CGO interface passes go pointers to the C API. This is an error by the CGO runtime. Disable the error by placing

```
export GODEBUG=cgocheck=0
```

in your `~/.bashrc` or `~/.zshrc` file and then run either `source ~/.bashrc` or `source ~/.zshrc`

### The MXNet C library

The MXNet C library is required.

If you use MXNet Docker Images (e.g. NVIDIA GPU CLOUD (NGC)), skip this step.

Refer to [go-mxnet](https://github.com/rai-project/go-mxnet#mxnet-installation) for mxnet installation.

## External services

Refer to [External services](https://github.com/rai-project/tensorflow#external-services).

## Use within MXNet Docker Images

Refer to [Use within TensorFlow Docker Images](https://github.com/rai-project/tensorflow#use-within-tensorflow-docker-images).

Continue if you have

* installed all the dependencies
* downloaded carml_config_example.yml to $HOME as .carml_config.yml
* launched docker external services on the host machine of the docker container you are going to use

, otherwise read above

An example of using NGC MXNet docker image: 

```
nvidia-docker run --shm-size=1g --ulimit memlock=-1 --ulimit stack=67108864 -it --privileged=true --network host \
-v $GOPATH:/workspace/go1.12/global \
-v $GOROOT:/workspace/go1.12_root \
-v ~/.carml_config.yml:/root/.carml_config.yml \
nvcr.io/nvidia/mxnet:19.06-py2
```

NOTE: The SHMEM allocation limit is set to the default of 64MB.  This may be
   insufficient for TensorFlow.  NVIDIA recommends the use of the following flags:
   ```nvidia-docker run --shm-size=1g --ulimit memlock=-1 --ulimit stack=67108864 ...```

Within the container, set up the environment so that the agent can find the TensorFlow C library.

```
export GOPATH=/workspace/go1.12/global
export GOROOT=/workspace/go1.12_root
export PATH=$GOROOT/bin:$PATH

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda/lib64:/usr/local/cuda/extras/CUPTI/lib64/
export CGO_LDFLAGS="${CGO_LDFLAGS} -L /usr/local/cuda/lib64 -L /usr/local/cuda/extras/CUPTI/lib64/"

export PATH=$PATH:$(go env GOPATH)/bin  
export GODEBUG=cgocheck=0  
cd $GOPATH/src/github.com/rai-project/mxnet/mxnet-agent  
```


## Usage

Refer to [Usage](https://github.com/rai-project/tensorflow#usage)
