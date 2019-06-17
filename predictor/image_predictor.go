package predictor

import (
	"context"
	"io/ioutil"

	opentracing "github.com/opentracing/opentracing-go"
	olog "github.com/opentracing/opentracing-go/log"
	"github.com/pkg/errors"
	"github.com/rai-project/dlframework"
	"github.com/rai-project/dlframework/framework/options"
	common "github.com/rai-project/dlframework/framework/predictor"
	"github.com/rai-project/downloadmanager"
	cupti "github.com/rai-project/go-cupti"
	gomxnet "github.com/rai-project/go-mxnet/mxnet"
	"github.com/rai-project/tracer"
	"gorgonia.org/tensor"
)

type ImagePredictor struct {
	common.ImagePredictor
	predictor *gomxnet.Predictor
}

func (p *ImagePredictor) Close() error {
	if p == nil {
		return nil
	}
	if p.predictor != nil {
		p.predictor.Close()
	}
	return nil
}

func (p *ImagePredictor) Load(ctx context.Context, model dlframework.ModelManifest, opts ...options.Option) (*ImagePredictor, error) {
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
				WorkDir:   workDir,
				Options:   options.New(opts...),
			},
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

// Download ...
func (p *ImagePredictor) Download(ctx context.Context, model dlframework.ModelManifest, opts ...options.Option) error {
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
				WorkDir:   workDir,
				Options:   options.New(opts...),
			},
		},
	}

	if err = ip.download(ctx); err != nil {
		return err
	}

	return nil
}

func (p *ImagePredictor) download(ctx context.Context) error {
	span, ctx := opentracing.StartSpanFromContext(
		ctx,
		"download",
		opentracing.Tags{
			"graph_url":           p.GetGraphUrl(),
			"target_graph_file":   p.GetGraphPath(),
			"weights_url":         p.GetWeightsUrl(),
			"target_weights_file": p.GetWeightsPath(),
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
	} else {
		span.LogFields(
			olog.String("event", "download model graph"),
		)
		_, err := downloadmanager.DownloadFile(
			p.GetGraphUrl(),
			p.GetGraphPath(),
			downloadmanager.MD5Sum(p.GetGraphChecksum()),
		)
		if err != nil {
			return err
		}

		span.LogFields(
			olog.String("event", "download model weights"),
		)
		_, err = downloadmanager.DownloadFile(
			p.GetWeightsUrl(),
			p.GetWeightsPath(),
			downloadmanager.MD5Sum(p.GetWeightsChecksum()),
		)
		if err != nil {
			return err
		}
	}

	if p.GetFeaturesUrl() != "" {
		span.LogFields(
			olog.String("event", "download features"),
		)
		_, err := downloadmanager.DownloadFile(
			p.GetFeaturesUrl(),
			p.GetFeaturesPath(),
			downloadmanager.MD5Sum(p.GetFeaturesChecksum()),
		)
		if err != nil {
			return err
		}
	}

	return nil
}

func (p *ImagePredictor) loadPredictor(ctx context.Context) error {
	if ctx != nil {
		span, _ := tracer.StartSpanFromContext(ctx, tracer.APPLICATION_TRACE, "load_predictor")
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

	opts, err := p.GetPredictionOptions()
	if err != nil {
		return err
	}

	inputLayer, err := p.GetInputLayerName("input_layer")
	if err != nil {
		return errors.Wrap(err, "failed to get the input layer name")
	}

	preprocessOpts, err := p.GetPreprocessOptions()
	if err != nil {
		return errors.Wrap(err, "failed to get the input preprocess options")
	}

	if err != nil {
		return errors.Wrap(err, "failed to get the input predict options")
	}
	var dtype tensor.Dtype
	switch t := preprocessOpts.ElementType; t {
	case "float32":
		dtype = tensor.Float32
	default:
		panic("currently only supports float32")
	}
	in := options.Node{
		Key:   inputLayer,
		Shape: append([]int{1}, preprocessOpts.Dims...),
		Dtype: dtype,
	}

	device := options.CPU_DEVICE
	if p.Options.UsesGPU() {
		device = options.CUDA_DEVICE
	}

	pred, err := gomxnet.New(
		ctx,
		options.WithOptions(opts),
		options.Device(device, 0),
		options.Graph(symbol),
		options.Weights(params),
		options.BatchSize(p.BatchSize()),
		options.InputNodes([]options.Node{in}),
	)

	if err != nil {
		return err
	}

	p.predictor = pred

	return nil
}

func (p *ImagePredictor) cuptiStart(ctx context.Context) (*cupti.CUPTI, error) {
	if !p.UseGPU() || p.TraceLevel() < tracer.HARDWARE_TRACE {
		return nil, nil
	}
	cu, err := cupti.New(cupti.Context(ctx), cupti.SamplingPeriod(0))
	if err != nil {
		return nil, err
	}
	return cu, nil
}

func (p *ImagePredictor) cuptiClose(cu *cupti.CUPTI) {
	if cu == nil {
		return
	}
	cu.Wait()
	cu.Close()
}
