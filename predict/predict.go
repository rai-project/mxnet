package predict

import (
	"bufio"
	"io/ioutil"
	"os"
	"path/filepath"
	"sort"
	"strings"

	"github.com/rai-project/tracer"

	context "context"

	opentracing "github.com/opentracing/opentracing-go"
	olog "github.com/opentracing/opentracing-go/log"
	"github.com/pkg/errors"
	"github.com/rai-project/config"
	"github.com/rai-project/dlframework"
	agent "github.com/rai-project/dlframework/framework/agent"
	"github.com/rai-project/dlframework/framework/feature"
	"github.com/rai-project/dlframework/framework/options"
	common "github.com/rai-project/dlframework/framework/predict"
	"github.com/rai-project/downloadmanager"
	gomxnet "github.com/rai-project/go-mxnet/mxnet"
	"github.com/rai-project/image"
	"github.com/rai-project/image/types"
	"github.com/rai-project/mxnet"
)

type ImagePredictor struct {
	common.ImagePredictor
	features  []string
	predictor *gomxnet.Predictor
}

func New(model dlframework.ModelManifest, opts ...options.Option) (common.Predictor, error) {
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

	predictor := new(ImagePredictor)

	return predictor.Load(ctx, model, opts...)
}

func (p *ImagePredictor) Download(ctx context.Context, model dlframework.ModelManifest, opts ...options.Option) error {
	predOpts := options.New(opts...)

	framework, err := model.ResolveFramework()
	if err != nil {
		return err
	}

	workDir, err := model.WorkDir()
	if err != nil {
		return err
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
		return err
	}

	return nil
}

func (p *ImagePredictor) Load(ctx context.Context, model dlframework.ModelManifest, opts ...options.Option) (common.Predictor, error) {
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
				Options:   options.New(opts...),
			},
			WorkDir: workDir,
		},
	}

	imageDims, err := p.GetImageDimensions()
	if err != nil {
		return err
	}

	ip.Options.Append(
		options.InputNode(ip.GetInputLayerName(DefaultInputLayerName), imageDims),
	)

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
		tracer.APPLICATION_TRACE,
		"download",
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
	span, ctx := tracer.StartSpanFromContext(ctx, tracer.APPLICATION_TRACE, "load_predictor")
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

	opts, err := p.GetPredictionOptions(ctx)
	if err != nil {
		return err
	}

	span.LogFields(
		olog.String("event", "creating predictor"),
	)

	pred, err := gomxnet.CreatePredictor(
		options.WithOptions(opts),
		options.Symbol(symbol),
		options.Weights(params),
	)
	if err != nil {
		return err
	}

	p.predictor = pred

	return nil
}

func (p *ImagePredictor) Predict(ctx context.Context, data [][]float32, opts ...options.Option) error {
	if p.TraceLevel() >= tracer.FRAMEWORK_TRACE {
		// define profiling options
		poptions := map[string]gomxnet.ProfileMode{
			"profile_all":        gomxnet.ProfileAllEnable,
			"profile_symbolic":   gomxnet.ProfileSymbolicOperatorsEnable,
			"profile_imperative": gomxnet.ProfileImperativeOperatorsEnable,
			"profile_memory":     gomxnet.ProfileMemoryEnable,
			"profile_api":        gomxnet.ProfileApiEnable,
			"continuous_dump":    gomxnet.ProfileContiguousDumpDisable,
			"dump_period":        gomxnet.ProfileDumpPeriod,
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

	var input []float32
	for _, v := range data {
		input = append(input, v...)
	}

	err := p.predictor.Predict(ctx, input)
	if err != nil {
		return err
	}

	return nil
}

// ReadPredictedFeatures ...
func (p *ImagePredictor) ReadPredictedFeatures(ctx context.Context) ([]dlframework.Features, error) {
	span, ctx := tracer.StartSpanFromContext(ctx, tracer.APPLICATION_TRACE, "read_predicted_features")
	defer span.Finish()

	probabilities, err := p.predictor.ReadPredictedOutput()
	if err != nil {
		return nil, err
	}

	var output []dlframework.Features
	batchSize := p.BatchSize()
	length := len(predictions) / batchSize

	for ii := 0; ii < batchSize; ii++ {
		rprobs := make([]*dlframework.Feature, length)
		for jj := 0; jj < length; jj++ {
			rprobs[jj] = feature.New(
				feature.ClassificationIndex(int32(jj)),
				feature.ClassificationName(p.features[jj]),
				feature.Probability(predictions[ii*length+jj].Probability),
			)
		}
		sort.Sort(dlframework.Features(rprobs))
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
