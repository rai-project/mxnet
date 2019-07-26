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
	"github.com/rai-project/mxnet"
	"github.com/rai-project/tracer"
	"gorgonia.org/tensor"
	gotensor "gorgonia.org/tensor"
)

type ImageClassificationPredictor struct {
	*ImagePredictor
	probabilitiesLayerIndex int
	probabilities           interface{}
}

func NewImageClassificationPredictor(model dlframework.ModelManifest, opts ...options.Option) (common.Predictor, error) {
	ctx := context.Background()
	span, ctx := tracer.StartSpanFromContext(ctx, tracer.APPLICATION_TRACE, "new_predictor")
	defer span.Finish()

	if tracer.GetLevel() >= tracer.FRAMEWORK_TRACE {
		// diable autotuning and batch inference
		opts = append(opts, options.DisableFrameworkAutoTuning(true))
	}

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

	pred.predictor.GetOptions().SetOutputNodes([]options.Node{
		options.Node{
			Dtype: tensor.Float32,
		},
	})

	p := &ImageClassificationPredictor{
		ImagePredictor: pred,
	}

	p.probabilitiesLayerIndex, err = p.GetOutputLayerIndex("probabilities_layer")
	if err != nil {
		return nil, errors.Wrap(err, "failed to get the probabilities layer index")
	}

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
	input, ok := data.([]*gotensor.Dense)
	if !ok {
		return errors.New("input data is not slice of go tensors")
	}

	err := p.cuptiStart(ctx)
	if err != nil {
		return err
	}

	err = p.predictor.Predict(ctx, input)
	if err != nil {
		return errors.Wrapf(err, "failed to perform Predict")
	}

	p.cuptiClose()

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

	return p.CreateClassificationFeatures(ctx, outputs[p.probabilitiesLayerIndex], labels)
}

func (p ImageClassificationPredictor) Modality() (dlframework.Modality, error) {
	return dlframework.ImageClassificationModality, nil
}

func init() {
	config.AfterInit(func() {
		framework := mxnet.FrameworkManifest
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