package ops

import (
	"fmt"

	"github.com/systemEng-Learning/go-ml-deployment/ir"
	"github.com/systemEng-Learning/go-ml-deployment/kernel"
	"github.com/systemEng-Learning/go-ml-deployment/tensor"
)

type Scaler struct {
	input  int
	offset []float32
	scale  []float32
	output int
}

func (s *Scaler) Init(k *kernel.Kernel, node *ir.NodeProto) error {
	input, err := k.RegisterReader(node.Input[0])
	if err != nil {
		return err
	}
	s.input = input
	for _, attr := range node.Attribute {
		switch attr.Name {
		case "offset":
			s.offset = attr.Floats
		case "scale":
			s.scale = attr.Floats
		default:
			return fmt.Errorf("%s not supported for %s", attr.Name, node.OpType)
		}
	}
	if len(s.scale) == 0 {
		return fmt.Errorf("scaler: scale attribute cannot be empty")
	}
	if len(s.scale) != len(s.offset) {
		return fmt.Errorf("scaler: scale length(%d) != offset length(%d)", len(s.scale), len(s.offset))
	}
	s.output = k.RegisterWriter(node.Output[0])
	return nil
}

func (s *Scaler) Compute(k *kernel.Kernel) error {
	data, err := k.Input(s.input)
	if err != nil {
		return err
	}
	input := data.Tensor
	if input.DType != tensor.Double && input.DType != tensor.Float && input.DType != tensor.Int32 && input.DType != tensor.Int64 {
		return fmt.Errorf("scaler: input datatype (%v) is invalid", input.DType)
	}

	output, err := k.Output(s.output, input.Shape, tensor.Float)
	if err != nil {
		return err
	}
	input.Cast(tensor.Float)
	rows := input.Shape[0]
	stride := rows
	if len(input.Shape) == 1 {
		rows = 1
	} else {
		stride = input.Shape[1]
	}
	length := rows * stride
	if len(s.offset) == stride {
		for i := range length {
			output.FloatData[i] = (input.FloatData[i] - s.offset[i%stride]) * s.scale[i%stride]
		}
	} else if len(s.offset) == 1 {
		for i := range length {
			output.FloatData[i] = (input.FloatData[i] - s.offset[0]) * s.scale[0]
		}
	} else {
		return fmt.Errorf("scaler: either offset/scale length has to be of length (%d) or 1", stride)
	}
	return nil
}
