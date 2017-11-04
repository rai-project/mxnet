package predict

import (
	"bufio"
	"io/ioutil"
	"os"
	"path/filepath"

	"github.com/rai-project/tracer"

	opentracing "github.com/opentracing/opentracing-go"
	olog "github.com/opentracing/opentracing-go/log"
	"github.com/pkg/errors"
	"github.com/rai-project/config"
	"github.com/rai-project/dlframework"
	agent "github.com/rai-project/dlframework/framework/agent"
	"github.com/rai-project/dlframework/framework/options"
	common "github.com/rai-project/dlframework/framework/predict"
	"github.com/rai-project/downloadmanager"
	gomxnet "github.com/rai-project/go-mxnet-predictor/mxnet"
	"github.com/rai-project/image"
	"github.com/rai-project/image/types"
	"github.com/rai-project/mxnet"
	context "golang.org/x/net/context"
)

type ImagePredictor struct {
	common.ImagePredictor
	features  []string
	predictor *gomxnet.Predictor
}

// func New(model dlframework.ModelManifest, opts ...options.Option) (common.Predictor, error) {
// 	modelInputs := model.GetInputs()
// 	if len(modelInputs) != 1 {
// 		return nil, errors.New("number of inputs not supported")
// 	}

// 	firstInputType := modelInputs[0].GetType()
// 	if strings.ToLower(firstInputType) != "image" {
// 		return nil, errors.New("input type not supported")
// 	}

// 	predictor := new(ImagePredictor)

// 	return predictor.Load(context.Background(), model, opts...)
// }

func (p *ImagePredictor) Load(ctx context.Context, model dlframework.ModelManifest, opts ...options.Option) (common.Predictor, error) {
	predOpts := options.New(opts...)
	span, ctx := tracer.StartSpanFromContext(ctx, tracer.STEP_TRACE, "Load")
	defer span.Finish()

	framework, err := model.ResolveFramework()
	if err != nil {
		return nil, err
	}

	workDir, err := model.WorkDir()
	if err != nil {
		return nil, err
	}

	ip := &ImagePredictor{
		ImagePredictor: common.ImagePredictor{
			Base: common.Base{
				Framework: framework,
				Model:     model,
				Options:   predOpts,
			},
			WorkDir: workDir,
		},
	}

	if err = ip.download(ctx); err != nil {
		return nil, err
	}

	if err = ip.loadPredictor(ctx); err != nil {
		return nil, err
	}

	return ip, nil
}

func (p *ImagePredictor) GetPreprocessOptions(ctx context.Context) (common.PreprocessOptions, error) {
	mean, err := p.GetMeanImage()
	if err != nil {
		return common.PreprocessOptions{}, err
	}

	scale, err := p.GetScale()
	if err != nil {
		return common.PreprocessOptions{}, err
	}

	imageDims, err := p.GetImageDimensions()
	if err != nil {
		return common.PreprocessOptions{}, err
	}

	return common.PreprocessOptions{
		Context:   ctx,
		MeanImage: mean,
		Scale:     scale,
		Size:      []int{int(imageDims[1]), int(imageDims[2])},
		ColorMode: types.RGBMode,
		Layout:    image.CHWLayout,
	}, nil
}

func (p *ImagePredictor) download(ctx context.Context) error {
	span, ctx := tracer.StartSpanFromContext(ctx,
		tracer.STEP_TRACE,
		"Download",
		opentracing.Tags{
			"graph_url":           p.GetGraphUrl(),
			"target_graph_file":   p.GetGraphPath(),
			"weights_url":         p.GetWeightsUrl(),
			"target_weights_file": p.GetWeightsPath(),
			"feature_url":         p.GetFeaturesUrl(),
			"target_feature_file": p.GetFeaturesPath(),
		},
	)
	defer span.Finish()

	model := p.Model
	if model.Model.IsArchive {
		baseURL := model.Model.BaseUrl
		span.LogFields(
			olog.String("event", "download model archive"),
		)
		_, err := downloadmanager.DownloadInto(baseURL, p.WorkDir, downloadmanager.Context(ctx))
		if err != nil {
			return errors.Wrapf(err, "failed to download model archive from %v", model.Model.BaseUrl)
		}
		return nil
	}
	checksum := p.GetGraphChecksum()
	if checksum == "" {
		return errors.New("Need graph file checksum in the model manifest")
	}

	span.LogFields(
		olog.String("event", "download graph"),
	)

	if _, err := downloadmanager.DownloadFile(p.GetGraphUrl(), p.GetGraphPath(), downloadmanager.MD5Sum(checksum)); err != nil {
		return err
	}

	checksum = p.GetWeightsChecksum()
	if checksum == "" {
		return errors.New("Need weights file checksum in the model manifest")
	}

	span.LogFields(
		olog.String("event", "download weights"),
	)
	if _, err := downloadmanager.DownloadFile(p.GetWeightsUrl(), p.GetWeightsPath(), downloadmanager.MD5Sum(checksum)); err != nil {
		return err
	}

	checksum = p.GetFeaturesChecksum()
	if checksum == "" {
		return errors.New("Need features file checksum in the model manifest")
	}

	span.LogFields(
		olog.String("event", "download features"),
	)
	if _, err := downloadmanager.DownloadFile(p.GetFeaturesUrl(), p.GetFeaturesPath(), downloadmanager.MD5Sum(checksum)); err != nil {
		return err
	}

	return nil
}

func (p *ImagePredictor) loadPredictor(ctx context.Context) error {
	span, ctx := tracer.StartSpanFromContext(ctx, tracer.STEP_TRACE, "LoadPredictor")
	defer span.Finish()

	span.LogFields(
		olog.String("event", "read graph"),
	)
	symbol, err := ioutil.ReadFile(p.GetGraphPath())
	if err != nil {
		return errors.Wrapf(err, "cannot read %s", p.GetGraphPath())
	}

	span.LogFields(
		olog.String("event", "read weights"),
	)
	params, err := ioutil.ReadFile(p.GetWeightsPath())
	if err != nil {
		return errors.Wrapf(err, "cannot read %s", p.GetWeightsPath())
	}

	span.LogFields(
		olog.String("event", "read features"),
	)
	var features []string
	f, err := os.Open(p.GetFeaturesPath())
	if err != nil {
		return errors.Wrapf(err, "cannot read %s", p.GetFeaturesPath())
	}
	defer f.Close()
	scanner := bufio.NewScanner(f)
	for scanner.Scan() {
		line := scanner.Text()
		features = append(features, line)
	}
	p.features = features

	inputDims, err := p.GetImageDimensions()
	if err != nil {
		return err
	}

	span.LogFields(
		olog.String("event", "creating predictor"),
	)

	opts, err := p.GetPredictionOptions(ctx)
	if err != nil {
		return err
	}

	pred, err := gomxnet.CreatePredictor(
		options.WithOptions(opts),
		options.Symbol(symbol),
		options.Weights(params),
		options.InputNode("data", inputDims),
	)
	if err != nil {
		return err
	}
	p.predictor = pred

	return nil
}

func (p *ImagePredictor) Predict(ctx context.Context, data [][]float32, opts ...options.Option) ([]dlframework.Features, error) {
	if p.TraceLevel() >= tracer.FRAMEWORK_TRACE {
		if profile, err := gomxnet.NewProfile(gomxnet.ProfileAllOperators, filepath.Join(p.WorkDir, "profile")); err == nil {
			profile.Start()
			defer func() {
				profile.Stop()
				profile.Publish(ctx)
				profile.Delete()
			}()
		}
	}

	var input []float32
	for _, v := range data {
		input = append(input, v...)
	}

	if err := p.predictor.SetInput("data", input); err != nil {
		return nil, err
	}

	if err := p.predictor.Forward(); err != nil {
		return nil, err
	}

	probabilities, err := p.predictor.GetOutput(0)
	if err != nil {
		return nil, err
	}

	var output []dlframework.Features
	batchSize := int(p.BatchSize())
	length := len(probabilities) / batchSize

	for i := 0; i < batchSize; i++ {
		rprobs := make([]*dlframework.Feature, length)
		for j := 0; j < length; j++ {
			rprobs[j] = &dlframework.Feature{
				Index:       int64(j),
				Name:        p.features[j],
				Probability: probabilities[i*length+j],
			}
		}
		output = append(output, rprobs)
	}

	return output, nil
}

func (p *ImagePredictor) Reset(ctx context.Context) error {

	return nil
}

func (p *ImagePredictor) Close() error {
	if p.predictor != nil {
		p.predictor.Close()
	}

	return nil
}

func init() {
	config.AfterInit(func() {
		framework := mxnet.FrameworkManifest
		agent.AddPredictor(framework, &ImagePredictor{
			ImagePredictor: common.ImagePredictor{
				Base: common.Base{
					Framework: framework,
				},
			},
		})
	})
}
