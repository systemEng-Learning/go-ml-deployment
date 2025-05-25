package ops

import (
	"fmt"

	"github.com/systemEng-Learning/go-ml-deployment/ir"
	"github.com/systemEng-Learning/go-ml-deployment/kernel"
	"github.com/systemEng-Learning/go-ml-deployment/tensor"
)
// binarizer struct - cast input? cast to? dtype of output tensor?
type Binarizer struct {
	input  int
	threshold float32
	output int
}

func (b *Binarizer) Init(k *kernel.Kernel, node *ir.NodeProto) error {
	input, err := k.RegisterReader(node.Input[0])
	if err != nil {
		return err
	}
	b.input = input
	
	b.threshold = 0.0
	for _, attr := range node.Attribute {
		if (attr.Name == "threshold") || (attr.Name == "t") {
			b.threshold = attr.F
		} else {
			return fmt.Errorf("%s not supported for %s", attr.Name, node.OpType)
		}
	}

	b.output = k.RegisterWriter(node.Output[0])
	return nil
}

func (b *Binarizer) Compute(k *kernel.Kernel) error {
	data, err := k.Input(b.input)
	if err != nil {
		return err
	}
	input := data.Tensor
	
	output, err := k.Output(b.output, input.Shape, input.DType)
	if err != nil {
		return err
	}

	switch input.DType {
	case tensor.Float:
		binarize(input.FloatData, output.FloatData, b.threshold)
	case tensor.Double:
		binarize(input.DoubleData, output.DoubleData, b.threshold)
	default:
		return fmt.Errorf("Binarizer only supports float32 and float64 tensors, got: %s", input.DType)
	}
	return nil
}

type Number interface {
	~float32 | ~float64
}
// function to apply threshold on input elements
func binarize[T Number](in []T, out []T, threshold float32) {
	for i, v := range in {
		if float32(v) > threshold {
			out[i] = 1
		} else {
			out[i] = 0
		}
	}
}