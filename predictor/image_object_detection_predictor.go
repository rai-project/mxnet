package predictor

import (
	"context"
	"strings"

	"github.com/pkg/errors"
	"github.com/rai-project/config"
	"github.com/rai-project/dlframework"
	"github.com/rai-project/dlframework/framework/agent"
	"github.com/rai-project/dlframework/framework/options"
	common "github.com/rai-project/dlframework/framework/predictor"
	"github.com/rai-project/mxnet"
	"github.com/rai-project/tracer"
	"gorgonia.org/tensor"
	gotensor "gorgonia.org/tensor"
)

type ObjectDetectionPredictor struct {
	*ImagePredictor
	classesLayerIndex       int
	classes                 interface{}
	probabilitiesLayerIndex int
	probabilities           interface{}
	boxesLayerIndex         int
	boxes                   interface{}
}

func NewObjectDetectionPredictor(model dlframework.ModelManifest, opts ...options.Option) (common.Predictor, error) {
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

	predictor := new(ObjectDetectionPredictor)

	return predictor.Load(ctx, model, opts...)
}

func (self *ObjectDetectionPredictor) Load(ctx context.Context, modelManifest dlframework.ModelManifest, opts ...options.Option) (common.Predictor, error) {
	pred, err := self.ImagePredictor.Load(ctx, modelManifest, opts...)
	if err != nil {
		return nil, err
	}

	pred.predictor.GetOptions().SetOutputNodes([]options.Node{
		options.Node{
			Dtype: tensor.Float32,
		},
		options.Node{
			Dtype: tensor.Float32,
		},
		options.Node{
			Dtype: tensor.Float32,
		},
	})

	p := &ObjectDetectionPredictor{
		ImagePredictor: pred,
	}

	p.classesLayerIndex, err = p.GetOutputLayerIndex("classes_layer")
	if err != nil {
		return nil, errors.Wrap(err, "failed to get the classes layer index")
	}

	p.probabilitiesLayerIndex, err = p.GetOutputLayerIndex("probabilities_layer")
	if err != nil {
		return nil, errors.Wrap(err, "failed to get the probabilities layer index")
	}

	p.boxesLayerIndex, err = p.GetOutputLayerIndex("boxes_layer")
	if err != nil {
		return nil, errors.Wrap(err, "failed to get the boxes layer index")
	}
	return p, nil
}

// Predict ...
func (p *ObjectDetectionPredictor) Predict(ctx context.Context, data interface{}, opts ...options.Option) error {
	span, ctx := tracer.StartSpanFromContext(ctx, tracer.APPLICATION_TRACE, "predict")
	defer span.Finish()

	if data == nil {
		return errors.New("input data nil")
	}
	input, ok := data.([]*gotensor.Dense)
	if !ok {
		return errors.New("input data is not slice of go tensors")
	}

	fst := input[0]
	joined, err := fst.Concat(0, input[1:]...)
	if err != nil {
		return errors.Wrap(err, "unable to concat tensors")
	}
	joined.Reshape(append([]int{len(input)}, fst.Shape()...)...)

	err = p.predictor.Predict(ctx, []*gotensor.Dense{joined})
	if err != nil {
		return errors.Wrapf(err, "failed to perform Predict")
	}

	return nil
}

// ReadPredictedFeatures ...
func (p *ObjectDetectionPredictor) ReadPredictedFeatures(ctx context.Context) ([]dlframework.Features, error) {
	span, ctx := tracer.StartSpanFromContext(ctx, tracer.APPLICATION_TRACE, "read_predicted_features")
	defer span.Finish()

	probabilities0, err := p.predictor.ReadPredictionOutputAtIndex(ctx, p.probabilitiesLayerIndex)
	if err != nil {
		return nil, err
	}
	probabilities, ok := probabilities0.Data().([]float32)
	if !ok {
		return nil, errors.New("probabilities is not of type []float32")
	}

	boxes0, err := p.predictor.ReadPredictionOutputAtIndex(ctx, p.boxesLayerIndex)
	if err != nil {
		return nil, err
	}
	boxes, ok := boxes0.Data().([]float32)
	if !ok {
		return nil, errors.New("boxes is not of type []float32")
	}

	classes0, err := p.predictor.ReadPredictionOutputAtIndex(ctx, p.classesLayerIndex)
	if err != nil {
		return nil, err
	}
	classes, ok := classes0.Data().([]float32)
	if !ok {
		return nil, errors.New("classes is not of type []float32")
	}

	labels, err := p.GetLabels()
	if err != nil {
		return nil, errors.New("cannot get the labels")
	}

	return p.CreateBoundingBoxFeatures(ctx, probabilities, classes, boxes, labels)
}

func (p ObjectDetectionPredictor) Modality() (dlframework.Modality, error) {
	return dlframework.ImageObjectDetectionModality, nil
}

func init() {
	config.AfterInit(func() {
		framework := mxnet.FrameworkManifest
		agent.AddPredictor(framework, &ObjectDetectionPredictor{
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
