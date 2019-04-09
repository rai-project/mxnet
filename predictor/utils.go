package predictor

// import (
// 	"os"

// 	"github.com/pkg/errors"
// 	gotensor "gorgonia.org/tensor"
// )

// func disableFrameworkAutoTuning() {
// 	os.Setenv("MXNET_CUDNN_AUTOTUNE_DEFAULT", "0")
// }

// func makeTensorFromGoTensors(in0 []*gotensor.Dense) ([]float32, error) {
// 	if len(in0) < 1 {
// 		return nil, errors.New("no dense tensor in input")
// 	}

// 	fst := in0[0]
// 	joined, err := fst.Concat(0, in0[1:]...)
// 	if err != nil {
// 		return nil, errors.Wrap(err, "unable to concat tensors")
// 	}
// 	joined.Reshape(append([]int{len(in0)}, fst.Shape()...)...)

// 	shape := make([]int64, len(joined.Shape()))
// 	for ii, s := range joined.Shape() {
// 		shape[ii] = int64(s)
// 	}

// 	switch t := in0[0].Dtype(); t {
// 	case gotensor.Uint8:
// 		return flattenedUint8ToTensor(joined.Data().([]uint8), shape)
// 	case gotensor.Uint16:
// 		return flattenedUint16ToTensor(joined.Data().([]uint16), shape)
// 	case gotensor.Uint32:
// 		return flattenedUint32ToTensor(joined.Data().([]uint32), shape)
// 	case gotensor.Int8:
// 		return flattenedInt8ToTensor(joined.Data().([]int8), shape)
// 	case gotensor.Int16:
// 		return flattenedInt16ToTensor(joined.Data().([]int16), shape)
// 	case gotensor.Int32:
// 		return flattenedInt32ToTensor(joined.Data().([]int32), shape)
// 	case gotensor.Float32:
// 		return flattenedFloat32ToTensor(joined.Data().([]float32), shape)
// 	case gotensor.Float64:
// 		return flattenedFloat64ToTensor(joined.Data().([]float64), shape)
// 	default:
// 		return nil, errors.Errorf("invalid element datatype %v", t)
// 	}
// }
