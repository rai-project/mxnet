package predictor

import (
	"os"
	"testing"

	"github.com/rai-project/config"
	_ "github.com/rai-project/tracer/jaeger"
)

func TestMain(m *testing.M) {
	config.Init(
		config.AppName("carml"),
		config.DebugMode(true),
		config.VerboseMode(true),
	)
	os.Exit(m.Run())
}
