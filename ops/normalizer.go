package ops

import (
	"fmt"

	"github.com/systemEng-Learning/go-ml-deployment/ir"
	"github.com/systemEng-Learning/go-ml-deployment/kernel"
	tensors "github.com/systemEng-Learning/go-ml-deployment/tensor"
)

type Normalizer struct {
	input  int
	output int
	norm   string
}

func (n *Normalizer) Init(k *kernel.Kernel, node *ir.NodeProto) error {
	input, err := k.RegisterReader(node.Input[0])
	if err != nil {
		return err
	}
	n.input = input
	for _, attr := range node.Attribute {
		switch attr.Name {
		case "norm":
			n.norm = string(attr.S)
		default:
			return fmt.Errorf("%s not supported for %s", attr.Name, node.OpType)
		}
	}
	n.output = k.RegisterWriter(node.Output[0])
	return nil
}

func (n *Normalizer) Compute(k *kernel.Kernel) error {
	data, err := k.Input(n.input)
	if err != nil {
		return err
	}
	original_dtype := data.Tensor.DType
	var input *tensors.Tensor
	if data.Readers == 1 {
		input = data.Tensor
	} else {
		input, err = data.Tensor.Clone()
		if err != nil {
			return err
		}
	}
	input.Cast(tensors.Double)
	var sum float64
	for i := range input.Shape[0] {
		sum = 0
		for j := range input.Shape[1] {
			sum += input.DoubleData[i*input.Shape[1]+j]
		}

		for j := range input.Shape[1] {
			input.DoubleData[i*input.Shape[1]+j] = input.DoubleData[i*input.Shape[1]+j] / sum
		}
	}
	input.Cast(original_dtype)
	err = k.Put(n.output, input)
	return err
}
