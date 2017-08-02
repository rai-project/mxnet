package agent

import (
	"fmt"
	"os"
	"strconv"
	"testing"
	"time"

	"github.com/facebookgo/freeport"
	"github.com/rai-project/config"
	dl "github.com/rai-project/dlframework"
	mxnet "github.com/rai-project/mxnet"
	"github.com/rai-project/utils"
	"github.com/stretchr/testify/assert"
)

func freePort() (string, error) {
	port, err := freeport.Get()
	if err != nil {
		return "", err
	}
	return strconv.Itoa(port), nil
}

func getAddress(port string) (string, error) {
	address, err := utils.GetLocalIp()
	if err != nil {
		return "", err
	}
	return fmt.Sprintf("%s:%s", address, port), nil
}

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
	port, err := freePort()
	assert.NoError(t, err)
	address, err := getAddress(port)
	assert.NoError(t, err)

	RegisterRegistryServer()
	RegisterPredictorServer(address)
	time.Sleep(3 * time.Second)
}

func TestMain(m *testing.M) {
	config.Init(
		config.AppName("carml"),
		config.DebugMode(true),
		config.VerboseMode(true),
	)
	os.Exit(m.Run())
}
