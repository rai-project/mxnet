package predict

import (
	"bufio"
	"image"
	"io/ioutil"
	"os"
	"strings"

	context "golang.org/x/net/context"

	"github.com/anthonynsimon/bild/parallel"
	opentracing "github.com/opentracing/opentracing-go"
	"github.com/pkg/errors"
	"github.com/rai-project/config"
	"github.com/rai-project/dlframework"
	agent "github.com/rai-project/dlframework/framework/agent"
	common "github.com/rai-project/dlframework/framework/predict"
	"github.com/rai-project/downloadmanager"
	gomxnet "github.com/rai-project/go-mxnet-predictor/mxnet"
	raiimage "github.com/rai-project/image"
	"github.com/rai-project/mxnet"
)

type ImagePredictor struct {
	common.ImagePredictor
	features  []string
	predictor *gomxnet.Predictor
}

func New(model dlframework.ModelManifest) (common.Predictor, error) {
	modelInputs := model.GetInputs()
	if len(modelInputs) != 1 {
		return nil, errors.New("number of inputs not supported")
	}

	firstInputType := modelInputs[0].GetType()
	if strings.ToLower(firstInputType) != "image" {
		return nil, errors.New("input type not supported")
	}

	predictor := new(ImagePredictor)

	return predictor.Load(context.Background(), model)
}

func (p *ImagePredictor) Load(ctx context.Context, model dlframework.ModelManifest) (common.Predictor, error) {
	if span, newCtx := opentracing.StartSpanFromContext(ctx, "Load"); span != nil {
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
				Framework: framework,
				Model:     model,
			},
			WorkDir: workDir,
		},
	}

	ip.download(ctx)
	ip.loadPredictor(ctx)

	return ip, nil
}

func (p *ImagePredictor) Preprocess(ctx context.Context, input interface{}) (interface{}, error) {
	if span, newCtx := opentracing.StartSpanFromContext(ctx, "Preprocess"); span != nil {
		ctx = newCtx
		defer span.Finish()
	}

	inputImage, ok := input.(image.Image)
	if !ok {
		return nil, errors.New("expecting an image input")
	}

	imageDims, err := p.GetImageDimensions()
	if err != nil {
		return nil, err
	}

	img, err := raiimage.Resize(ctx, inputImage, int(imageDims[2]), int(imageDims[3]))
	if err != nil {
		return nil, errors.Wrap(err, "failed to resize input image")
	}
	b := img.Bounds()
	height := b.Max.Y - b.Min.Y // image height
	width := b.Max.X - b.Min.X  // image width

	mean, err := p.GetMeanImage(ctx)
	if err != nil {
		return nil, errors.Wrap(err, "failed to get mean image")
	}

	scale, err := p.GetScale()
	if err != nil {
		return nil, errors.Wrap(err, "failed to get scale")
	}

	res := make([]float32, 3*height*width)
	parallel.Line(height, func(start, end int) {
		w := width
		h := height
		for y := start; y < end; y++ {
			for x := 0; x < width; x++ {
				r, g, b, _ := img.At(x+b.Min.X, y+b.Min.Y).RGBA()
				res[y*w+x] = (float32(r>>8) - mean[0]) / scale
				res[w*h+y*w+x] = (float32(g>>8) - mean[1]) / scale
				res[2*w*h+y*w+x] = (float32(b>>8) - mean[2]) / scale
			}
		}
	})

	return res, nil
}

func (p *ImagePredictor) download(ctx context.Context) error {
	if span, newCtx := opentracing.StartSpanFromContext(ctx, "Download"); span != nil {
		span.SetTag("graph_url", p.GetGraphUrl())
		span.SetTag("traget_graph_file", p.GetGraphPath())
		span.SetTag("weights_url", p.GetWeightsUrl())
		span.SetTag("traget_weights_file", p.GetWeightsPath())
		span.SetTag("feature_url", p.GetFeaturesUrl())
		span.SetTag("traget_feature_file", p.GetFeaturesPath())
		ctx = newCtx
		defer span.Finish()
	}

	model := p.Model
	if model.Model.IsArchive {
		baseURL := model.Model.BaseUrl
		_, err := downloadmanager.DownloadInto(baseURL, p.WorkDir)
		if err != nil {
			return errors.Wrapf(err, "failed to download model archive from %v", model.Model.BaseUrl)
		}
		return nil
	}

	if _, err := downloadmanager.DownloadFile(p.GetGraphUrl(), p.GetGraphPath()); err != nil {
		return err
	}

	if _, err := downloadmanager.DownloadFile(p.GetWeightsUrl(), p.GetWeightsPath()); err != nil {
		return err
	}

	if _, err := downloadmanager.DownloadFile(p.GetFeaturesUrl(), p.GetFeaturesPath()); err != nil {
		return err
	}

	return nil
}

func (p *ImagePredictor) loadPredictor(ctx context.Context) error {
	if span, newCtx := opentracing.StartSpanFromContext(ctx, "LoadPredictor"); span != nil {
		ctx = newCtx
		defer span.Finish()
	}

	symbol, err := ioutil.ReadFile(p.GetGraphPath())
	if err != nil {
		return errors.Wrapf(err, "cannot read %s", p.GetGraphPath())
	}

	params, err := ioutil.ReadFile(p.GetWeightsPath())
	if err != nil {
		return errors.Wrapf(err, "cannot read %s", p.GetWeightsPath())
	}

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

	pred, err := gomxnet.CreatePredictor(symbol,
		params,
		gomxnet.Device{gomxnet.CPU_DEVICE, 0},
		[]gomxnet.InputNode{{Key: "data", Shape: inputDims}},
	)
	if err != nil {
		return err
	}
	p.predictor = pred

	return nil
}

func (p *ImagePredictor) Predict(ctx context.Context, data []float32) (dlframework.Features, error) {
	if span, newCtx := opentracing.StartSpanFromContext(ctx, "Predict"); span != nil {
		ctx = newCtx
		defer span.Finish()
	}

	if err := p.predictor.SetInput("data", data); err != nil {
		return nil, err
	}

	if err := p.predictor.Forward(); err != nil {
		return nil, err
	}

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
