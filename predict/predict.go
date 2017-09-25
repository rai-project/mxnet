package predict

import (
	"bufio"
	"io/ioutil"
	"os"
	"path/filepath"
	"strings"

	opentracing "github.com/opentracing/opentracing-go"
	olog "github.com/opentracing/opentracing-go/log"
	"github.com/pkg/errors"
	"github.com/rai-project/config"
	"github.com/rai-project/dlframework"
	agent "github.com/rai-project/dlframework/framework/agent"
	common "github.com/rai-project/dlframework/framework/predict"
	"github.com/rai-project/downloadmanager"
	gomxnet "github.com/rai-project/go-mxnet-predictor/mxnet"
	"github.com/rai-project/image/types"
	"github.com/rai-project/mxnet"
	context "golang.org/x/net/context"
)

type ImagePredictor struct {
	common.ImagePredictor
	features  []string
	predictor *gomxnet.Predictor
}

func New(model dlframework.ModelManifest, opts dlframework.PredictionOptions) (common.Predictor, error) {
	modelInputs := model.GetInputs()
	if len(modelInputs) != 1 {
		return nil, errors.New("number of inputs not supported")
	}

	firstInputType := modelInputs[0].GetType()
	if strings.ToLower(firstInputType) != "image" {
		return nil, errors.New("input type not supported")
	}

	predictor := new(ImagePredictor)

	return predictor.Load(context.Background(), model, opts)
}

func (p *ImagePredictor) Load(ctx context.Context, model dlframework.ModelManifest, opts dlframework.PredictionOptions) (common.Predictor, error) {
	if span, newCtx := tracer.StartSpanFromContext(ctx, "Load"); span != nil {
		ctx = newCtx
		defer span.Finish()
	}

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
				Framework:         framework,
				Model:             model,
				PredictionOptions: opts,
				Tracer:            tracer,
			},
			WorkDir: workDir,
		},
	}

	if ip.download(ctx) != nil {
		return nil, err
	}

	if ip.loadPredictor(ctx) != nil {
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
		MeanImage: mean,
		Scale:     scale,
		Size:      []int{int(imageDims[1]), int(imageDims[2])},
		ColorMode: types.RGBMode,
	}, nil
}

func (p *ImagePredictor) download(ctx context.Context) error {
	span, ctx := p.GetTracer().StartSpanFromContext(
		ctx,
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
	span, ctx := p.GetTracer().StartSpanFromContext(ctx, "LoadPredictor")

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
	pred, err := gomxnet.CreatePredictor(
		gomxnet.Symbol(symbol),
		gomxnet.Weights(params),
		gomxnet.InputNode("data", inputDims),
		gomxnet.BatchSize(p.BatchSize()),
	)
	if err != nil {
		return err
	}
	p.predictor = pred

	return nil
}

func (p *ImagePredictor) Predict(ctx context.Context, data []float32, opts dlframework.PredictionOptions) (dlframework.Features, error) {
	span, ctx := p.GetTracer().StartSpanFromContext(ctx, "Predict", opentracing.Tags{
		"model_name":        p.Model.GetName(),
		"model_version":     p.Model.GetVersion(),
		"framework_name":    p.Model.GetFramework().GetName(),
		"framework_version": p.Model.GetFramework().GetVersion(),
	})
	if profile, err := gomxnet.NewProfile(gomxnet.ProfileAllOperators, filepath.Join(p.WorkDir, "profile")); err == nil {
		profile.Start()
		defer func() {
			profile.Stop()
			profile.Publish(ctx, "layers")
			profile.Delete()
		}()
	}
	defer span.Finish()

	span.LogFields(
		olog.String("event", "setting input"),
	)
	if err := p.predictor.SetInput("data", data); err != nil {
		return nil, err
	}

	span.LogFields(
		olog.String("event", "forward"),
	)
	if err := p.predictor.Forward(); err != nil {
		return nil, err
	}

	span.LogFields(
		olog.String("event", "getting output"),
	)
	probabilities, err := p.predictor.GetOutput(0)
	if err != nil {
		return nil, err
	}

	rprobs := make([]*dlframework.Feature, len(probabilities))
	for ii, prob := range probabilities {
		rprobs[ii] = &dlframework.Feature{
			Index:       int64(ii),
			Name:        p.features[ii],
			Probability: prob,
		}
	}
	res := dlframework.Features(rprobs)

	return res, nil
}

func (p *ImagePredictor) Reset(ctx context.Context) error {

	return nil
}

func (p *ImagePredictor) Close() error {
	if p.predictor != nil {
		p.predictor.Free()
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
