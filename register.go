package mxnet

import (
	"os"

	assetfs "github.com/elazarl/go-bindata-assetfs"
	"github.com/rai-project/dlframework"
	"github.com/rai-project/dlframework/framework"
)

var FrameworkManifest = dlframework.FrameworkManifest{
	Name:    "MXNet",
	Version: "0.1",
	Container: map[string]*dlframework.ContainerHardware{
		"amd64": {
			Cpu: "raiproject/carml-mxnet:amd64-cpu",
			Gpu: "raiproject/carml-mxnet:amd64-gpu",
		},
		"ppc64le": {
			Cpu: "raiproject/carml-mxnet:ppc64le-gpu",
			Gpu: "raiproject/carml-mxnet:ppc64le-gpu",
		},
	},
}

func assetFS() *assetfs.AssetFS {
	assetInfo := func(path string) (os.FileInfo, error) {
		return os.Stat(path)
	}
	for k := range _bintree.Children {
		return &assetfs.AssetFS{Asset: Asset, AssetDir: AssetDir, AssetInfo: assetInfo, Prefix: k}
	}
	panic("unreachable")
}

func init() {
	framework.Register(FrameworkManifest, assetFS())
}
