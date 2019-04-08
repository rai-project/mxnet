package main

import (
	"fmt"
	"os"

	cmd "github.com/rai-project/dlframework/framework/cmd/server"
	"github.com/rai-project/mxnet"
	_ "github.com/rai-project/mxnet/predictor"
)

func main() {
	rootCmd, err := cmd.NewRootCommand(mxnet.Register, mxnet.FrameworkManifest)
	if err != nil {
		fmt.Println(err)
		os.Exit(-1)
	}

	if err := rootCmd.Execute(); err != nil {
		fmt.Println(err)
		os.Exit(-1)
	}
}
