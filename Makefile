all: generate

fmt:
	go fmt ./...

install-deps:
	go get github.com/jteeuwen/go-bindata/...
	go get github.com/elazarl/go-bindata-assetfs/...
	go get github.com/golang/dep
	dep ensure -v

generate: clean generate-models

generate-proto:
	protoc --gogofaster_out=Mgoogle/protobuf/any.proto=github.com/gogo/protobuf/types,plugins=grpc:. -Iproto -I$(GOPATH)/src proto/mxnet.proto

generate-models:
	go-bindata -nomemcopy -prefix builtin_models/ -pkg mxnet -o builtin_models_static.go -ignore=.DS_Store  -ignore=README.md builtin_models/...

clean-models:
	rm -fr builtin_models_static.go

clean-proto:
	rm -fr *pb.go

clean: clean-models

travis: install-deps generate
	echo "building..."
	go build
