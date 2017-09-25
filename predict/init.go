package predict

import (
	"github.com/rai-project/config"
	"github.com/rai-project/logger"
	tr "github.com/rai-project/tracer"
	"github.com/sirupsen/logrus"
)

var (
	log    *logrus.Entry
	tracer tr.Tracer
)

func init() {
	config.AfterInit(func() {
		log = logger.New().WithField("pkg", "mxnet/predict")
		t, err := tr.New("mxnet")
		if err != nil {
			// just use the noop tracer
			t, err = tr.NewFromName("mxnet", "noop")
			if err != nil {
				panic(err)
			}
		}
		tracer = t
	})
}
