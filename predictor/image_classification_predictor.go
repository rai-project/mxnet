package predictor

import (
	"context"
	"path/filepath"
	"strings"

	"github.com/pkg/errors"
	"github.com/rai-project/config"
	"github.com/rai-project/dlframework"
	"github.com/rai-project/dlframework/framework/agent"
	"github.com/rai-project/dlframework/framework/options"
	common "github.com/rai-project/dlframework/framework/predictor"
	gomxnet "github.com/rai-project/go-mxnet/mxnet"
	"github.com/rai-project/tensorflow"
	"github.com/rai-project/tracer"
	gotensor "gorgonia.org/tensor"
)

type ImageClassificationPredictor struct {
	*ImagePredictor
	inputLayer              string
	probabilitiesLayerIndex int
}

func NewImageClassificationPredictor(model dlframework.ModelManifest, opts ...options.Option) (common.Predictor, error) {
	ctx := context.Background()
	span, ctx := tracer.StartSpanFromContext(ctx, tracer.APPLICATION_TRACE, "new_predictor")
	defer span.Finish()

	modelInputs := model.GetInputs()
	if len(modelInputs) != 1 {
		return nil, errors.New("number of inputs not supported")
	}
	firstInputType := modelInputs[0].GetType()
	if strings.ToLower(firstInputType) != "image" {
		return nil, errors.New("input type not supported")
	}

	predictor := new(ImageClassificationPredictor)
	return predictor.Load(ctx, model, opts...)
}

func (self *ImageClassificationPredictor) Load(ctx context.Context, modelManifest dlframework.ModelManifest, opts ...options.Option) (common.Predictor, error) {

	pred, err := self.ImagePredictor.Load(ctx, modelManifest, opts...)
	if err != nil {
		return nil, err
	}

	p := &ImageClassificationPredictor{
		ImagePredictor: pred,
	}

	// symbol, err := ioutil.ReadFile(p.GetGraphPath())
	// if err != nil {
	// 	return nil, errors.Wrapf(err, "cannot read %s", p.GetGraphPath())
	// }

	// params, err := ioutil.ReadFile(p.GetWeightsPath())
	// if err != nil {
	// 	return nil, errors.Wrapf(err, "cannot read %s", p.GetWeightsPath())
	// }

	// p.options.SetGraph(symbol)
	// p.options.SetGraph(symbol)
	// p.probabilitiesLayerIndex, err = p.GetOutputLayerIndex("probabilities_layer")
	// if err != nil {
	// 	return nil, errors.Wrap(err, "failed to get the probabilities layer name")
	// }

	return p, nil
}

// Predict ...
func (p *ImageClassificationPredictor) Predict(ctx context.Context, data interface{}, opts ...options.Option) error {
	span, ctx := tracer.StartSpanFromContext(ctx, tracer.APPLICATION_TRACE, "predict")
	defer span.Finish()

	if p.TraceLevel() >= tracer.FRAMEWORK_TRACE {
		// define profiling options
		poptions := map[string]gomxnet.ProfileMode{
			"profile_all":        gomxnet.ProfileAllEnable,
			"profile_symbolic":   gomxnet.ProfileSymbolicOperatorsDisable,
			"profile_imperative": gomxnet.ProfileImperativeOperatorsDisable,
			"profile_memory":     gomxnet.ProfileMemoryDisable,
			"profile_api":        gomxnet.ProfileApiDisable,
			"continuous_dump":    gomxnet.ProfileContinuousDumpDisable,
		}
		if profile, err := gomxnet.NewProfile(poptions, filepath.Join(p.WorkDir, "profile")); err == nil {
			profile.Start()
			defer func() {
				profile.Stop()
				profile.Publish(context.WithValue(ctx, "graph_path", p.GetGraphPath()))
				profile.Delete()
			}()
		}
	}

	if data == nil {
		return errors.New("input data nil")
	}
	input, ok := data.([]gotensor.Tensor)
	if !ok {
		return errors.New("input data is not slice of dense tensors")
	}

	cu, err := p.cuptiStart(ctx)
	if err != nil {
		return err
	}

	err = p.predictor.Predict(ctx, input)
	if err != nil {
		return errors.Wrapf(err, "failed to perform Predict")
	}

	p.cuptiClose(cu)

	return nil
}

// ReadPredictedFeatures ...
func (p *ImageClassificationPredictor) ReadPredictedFeatures(ctx context.Context) ([]dlframework.Features, error) {
	span, ctx := tracer.StartSpanFromContext(ctx, tracer.APPLICATION_TRACE, "read_predicted_features")
	defer span.Finish()

	outputs, err := p.predictor.ReadPredictionOutputs(ctx)
	if err != nil {
		return nil, err
	}

	labels, err := p.GetLabels()
	if err != nil {
		return nil, errors.New("cannot get the labels")
	}

	return p.CreateClassificationFeatures(ctx, outputs[0], labels)
}

func (p ImageClassificationPredictor) Modality() (dlframework.Modality, error) {
	return dlframework.ImageClassificationModality, nil
}

func init() {
	config.AfterInit(func() {
		framework := tensorflow.FrameworkManifest
		agent.AddPredictor(framework, &ImageClassificationPredictor{
			ImagePredictor: &ImagePredictor{
				ImagePredictor: common.ImagePredictor{
					Base: common.Base{
						Framework: framework,
					},
				},
			},
		})
	})
}
