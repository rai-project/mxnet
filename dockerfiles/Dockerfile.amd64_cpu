FROM carml/go-mxnet:amd64-cpu-latest
MAINTAINER Abdul Dakkak <dakkak@illinois.edu>


 # Build-time metadata as defined at http://label-schema.org
ARG BUILD_DATE
ARG VCS_REF
ARG VCS_URL
ARG VERSION
ARG ARCH
LABEL org.mlmodelscope.mxnet.agent.build-date=$BUILD_DATE \
      org.mlmodelscope.mxnet.agent.name="mxnet agent" \
      org.mlmodelscope.mxnet.agent.description="" \
      org.mlmodelscope.mxnet.agent.url="https://www.mlmodelscope.org/" \
      org.mlmodelscope.mxnet.agent.vcs-ref=$VCS_REF \
      org.mlmodelscope.mxnet.agent.vcs-url=$VCS_URL \
      org.mlmodelscope.mxnet.agent.vendor="MLModelScope" \
      org.mlmodelscope.mxnet.agent.version=$VERSION \
      org.mlmodelscope.mxnet.agent.schema-version="1.0"


# Get Go bindings
ENV PKG github.com/rai-project/mxnet
WORKDIR $GOPATH/src/$PKG

RUN git clone --depth=1 https://${PKG}.git .

RUN dep ensure -v -no-vendor -update \
    github.com/rai-project/mxnet \
    github.com/rai-project/go-mxnet \
    github.com/rai-project/dlframework \
    github.com/rai-project/evaluation && \
    dep ensure -v -vendor-only

RUN go build -tags=nogpu -a -installsuffix cgo -ldflags "-s -w -X ${PKG}/Version=${VERSION} -X ${PKG}/GitCommit=${VCS_REF} -X ${PKG}/BuildDate=${BUILD_DATE}"&& \
    go install -tags=nogpu && \
    rm -fr vendor
