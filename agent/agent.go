package agent

import (
	"image"
	_ "image/gif"
	_ "image/jpeg"
	_ "image/png"

	"google.golang.org/grpc"

	opentracing "github.com/opentracing/opentracing-go"
	"github.com/pkg/errors"
	rgrpc "github.com/rai-project/grpc"
	"github.com/rai-project/mxnet"
	"github.com/rai-project/registry"
	context "golang.org/x/net/context"

	dl "github.com/rai-project/dlframework"
	"github.com/rai-project/utils"

	"github.com/rai-project/uuid"

	common "github.com/rai-project/dlframework/framework/agent"
	predict "github.com/rai-project/mxnet/predict"
)

type registryServer struct {
	common.Registry
}

type predictorServer struct {
	common.Predictor
}

func (p *predictorServer) Predict(ctx context.Context, req *dl.PredictRequest) (*dl.PredictResponse, error) {
	_, model, err := p.FindFrameworkModel(ctx, req)
	if err != nil {
		return nil, err
	}

	predictor, err := predict.New(*model)
	if err != nil {
		return nil, err
	}
	defer predictor.Close()
	if err := predictor.Download(ctx); err != nil {
		return nil, err
	}

	reader, err := p.InputReaderCloser(ctx, req)
	if err != nil {
		return nil, err
	}
	defer reader.Close()

	var img image.Image

	func() {
		if span, newCtx := opentracing.StartSpanFromContext(ctx, "DecodeImage"); span != nil {
			ctx = newCtx
			defer span.Finish()
		}
		img, _, err = image.Decode(reader)
	}()
	if err != nil {
		return nil, errors.Wrapf(err, "unable to read input as image")
	}

	data, err := predictor.Preprocess(ctx, img)
	if err != nil {
		return nil, err
	}

	probs, err := predictor.Predict(ctx, data)
	if err != nil {
		return nil, err
	}

	probs.Sort()

	if req.GetLimit() != 0 {
		trunc := probs.Take(int(req.GetLimit()))
		probs = &trunc
	}

	return &dl.PredictResponse{
		Id:       uuid.NewV4(),
		Features: *probs,
		Error:    nil,
	}, nil
}

func RegisterRegistryServer() (*grpc.Server, error) {
	log.Info("populating registry")

	var grpcServer *grpc.Server
	grpcServer = rgrpc.NewServer(dl.RegistryServiceDescription)
	svr := &registryServer{
		Registry: common.Registry{
			Base: common.Base{
				Framework: mxnet.FrameworkManifest,
			},
		},
	}
	go func() {
		utils.Every(
			registry.Config.Timeout/2,
			func() {
				svr.PublishInRegistery()
			},
		)
	}()
	dl.RegisterRegistryServer(grpcServer, svr)
	return grpcServer, nil
}

func RegisterPredictorServer(host string) (*grpc.Server, error) {
	log.Info("registering predictor service at ", host)

	grpcServer := rgrpc.NewServer(dl.PredictorServiceDescription)

	svr := &predictorServer{
		Predictor: common.Predictor{
			Host: host,
			Base: common.Base{
				Framework: mxnet.FrameworkManifest,
			},
		},
	}
	go func() {
		utils.Every(
			registry.Config.Timeout/2,
			func() {
				svr.PublishInRegistery()
			},
		)
	}()
	dl.RegisterPredictorServer(grpcServer, svr)
	return grpcServer, nil
}
