package ops

import (
	"errors"
	"fmt"

	"github.com/systemEng-Learning/go-ml-deployment/ir"
	"github.com/systemEng-Learning/go-ml-deployment/kernel"
	tensors "github.com/systemEng-Learning/go-ml-deployment/tensor"
)

type Cast struct {
	input    int
	output   int
	to       tensors.DataType
	saturate bool
}

func (c *Cast) Init(k *kernel.Kernel, node *ir.NodeProto) error {
	input, err := k.RegisterReader(node.Input[0])
	if err != nil {
		return err
	}
	c.input = input
	c.saturate = true
	for _, attr := range node.Attribute {
		switch attr.Name {
		case "to":
			c.to = tensors.OnnxTypeToDtype(int32(attr.I))
		case "saturate":
			if attr.I == 0 {
				c.saturate = false
			}
		default:
			return fmt.Errorf("%s not supported for %s", attr.Name, node.OpType)
		}
	}
	c.output = k.RegisterWriter(node.Output[0])
	return nil
}

func (c *Cast) Compute(k *kernel.Kernel) error {
	data, err := k.Input(c.input)
	input := data.Tensor
	if err != nil {
		return err
	}
	if input.DType != c.to {
		return errors.ErrUnsupported
	}
	if data.Readers == 1 {
		// Just place input in output
		err = k.Put(c.output, input)
	} else {
		// We are not the only reader, so clone
		var output *tensors.Tensor
		output, err = input.Clone()
		if err != nil {
			return err
		}
		err = k.Put(c.output, output)
	}

	return err
}
