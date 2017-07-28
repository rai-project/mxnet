package agent

import (
	"fmt"
	"os"
	"testing"

	"github.com/rai-project/config"
	dl "github.com/rai-project/dlframework"
	mxnet "github.com/rai-project/mxnet"
	"github.com/stretchr/testify/assert"
)

func XXXTestFrameworkRegistration(t *testing.T) {
	framework := mxnet.FrameworkManifest
	models := framework.Models()
	assert.NotEmpty(t, models)
}

func XXXTestModelRegistration(t *testing.T) {
	models, err := dl.Models()
	assert.NoError(t, err)
	for _, model := range models {
		fmt.Println(model.GetName() + ":" + model.GetVersion())
	}
}

func TestGRPCRegistration(t *testing.T) {
	RegisterRegistryServer()
	RegisterPredictorServer()
}

func TestMain(m *testing.M) {
	config.Init(
		config.AppName("carml"),
		config.DebugMode(true),
		config.VerboseMode(true),
	)
	os.Exit(m.Run())
}
