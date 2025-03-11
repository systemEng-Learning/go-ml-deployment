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
	input, err := k.GetTensorIndex(node.Input[0])
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
	n.output = k.RegisterTensor(node.Output[0])
	return nil
}

func (n *Normalizer) Compute(k *kernel.Kernel) error {
	input, err := k.Input(n.input)
	if err != nil {
		return err
	}
	cloned, err := input.Clone()
	if err != nil {
		return err
	}
	cloned.Cast(tensors.Double)
	var sum float64
	for i := range cloned.Shape[0] {
		sum = 0
		for j := range cloned.Shape[1] {
			sum += cloned.DoubleData[i*cloned.Shape[1]+j]
		}

		for j := range cloned.Shape[1] {
			cloned.DoubleData[i*cloned.Shape[1]+j] = cloned.DoubleData[i*cloned.Shape[1]+j] / sum
		}
	}
	cloned.Cast(input.DType)
	err = k.Put(n.output, cloned)
	return err
}
