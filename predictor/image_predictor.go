package predictor

import (
	"context"
	"io"
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
)

type ImagePredictor struct {
	common.ImagePredictor
	predictor *gomxnet.Predictor
}

func (p *ImagePredictor) Close() error {
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

	if ip.Options.DisableFrameworkAutoTuning() {
		disableFrameworkAutoTuning()
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
		if _, err := downloadmanager.DownloadInto(baseURL, p.WorkDir, downloadmanager.Context(ctx)); err != nil {
			return errors.Wrapf(err, "failed to download model archive from %v", model.Model.BaseUrl)
		}
	} else {
		span.LogFields(
			olog.String("event", "download model graph"),
		)
		checksum := p.GetGraphChecksum()
		if checksum != "" {
			if _, err := downloadmanager.DownloadFile(p.GetGraphUrl(), p.GetGraphPath(), downloadmanager.MD5Sum(checksum)); err != nil {
				return err
			}
		} else {
			if _, err := downloadmanager.DownloadFile(p.GetGraphUrl(), p.GetGraphPath()); err != nil {
				return err
			}
    }
    
    span.LogFields(
			olog.String("event", "download model weights"),
		)
		checksum = p.GetWeightsChecksum()
		if checksum != "" {
			if _, err := downloadmanager.DownloadFile(p.GetWeightsUrl(), p.GetWeightsPath(), downloadmanager.MD5Sum(checksum)); err != nil {
				return err
			}
		} else {
			if _, err := downloadmanager.DownloadFile(p.GetWeightsUrl(), p.GetWeightsPath()); err != nil {
				return err
			}
		}
	}

	if p.GetFeaturesUrl() != "" {
		span.LogFields(
			olog.String("event", "download features"),
		)
		checksum := p.GetFeaturesChecksum()
		if checksum != "" {
			if _, err := downloadmanager.DownloadFile(p.GetFeaturesUrl(), p.GetFeaturesPath(), downloadmanager.MD5Sum(checksum)); err != nil {
				return err
			}
		} else {
			if _, err := downloadmanager.DownloadFile(p.GetFeaturesUrl(), p.GetFeaturesPath()); err != nil {
				return err
			}
		}
	}

	return nil
}

func (p *ImagePredictor) loadPredictor(ctx context.Context) error {
	if ctx != nil {
		span, _ := tracer.StartSpanFromContext(ctx, tracer.APPLICATION_TRACE, "load_predictor")
		defer span.Finish()
		span.LogFields(
			olog.String("event", "read graph"),
		)
	}

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

	opts, err := p.GetPredictionOptions(ctx)
	if err != nil {
		return err
	}

	span.LogFields(
		olog.String("event", "creating predictor"),
	)

	pred, err := gomxnet.New(
		ctx,
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
