# MLModelScope MXNet Agent

[![Build Status](https://travis-ci.org/rai-project/mxnet.svg?branch=master)](https://travis-ci.org/rai-project/mxnet)
[![Build Status](https://dev.azure.com/dakkak/rai/_apis/build/status/mxnet)](https://dev.azure.com/dakkak/rai/_build/latest?definitionId=17)
[![Go Report Card](https://goreportcard.com/badge/github.com/rai-project/mxnet)](https://goreportcard.com/report/github.com/rai-project/mxnet)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

[![](https://images.microbadger.com/badges/version/carml/mxnet:ppc64le-gpu-latest.svg)](https://microbadger.com/images/carml/mxnet:ppc64le-gpu-latest> 'Get your own version badge on microbadger.com') [![](https://images.microbadger.com/badges/version/carml/mxnet:ppc64le-cpu-latest.svg)](https://microbadger.com/images/carml/mxnet:ppc64le-cpu-latest 'Get your own version badge on microbadger.com') [![](https://images.microbadger.com/badges/version/carml/mxnet:amd64-cpu-latest.svg)](https://microbadger.com/images/carml/mxnet:amd64-cpu-latest 'Get your own version badge on microbadger.com') [![](https://images.microbadger.com/badges/version/carml/mxnet:amd64-gpu-latest.svg)](https://microbadger.com/images/carml/mxnet:amd64-gpu-latest 'Get your own version badge on microbadger.com')


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

## Usage

Refer to [Usage](https://github.com/rai-project/tensorflow#usage)
