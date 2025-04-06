package ops

import (
	"fmt"

	"github.com/systemEng-Learning/go-ml-deployment/ir"
	"github.com/systemEng-Learning/go-ml-deployment/kernel"
	"github.com/systemEng-Learning/go-ml-deployment/tensor"
)

type LinearRegressor struct {
	input        int
	outputs      []int
	coefficients *tensor.Tensor
	intercepts   *tensor.Tensor
}

func (l *LinearRegressor) Init(k *kernel.Kernel, node *ir.NodeProto) error {
	input, err := k.RegisterReader(node.Input[0])
	if err != nil {
		return err
	}

	l.input = input
	for _, attr := range node.Attribute {
		switch attr.Name {
		case "coefficients":
			l.coefficients = tensor.Create1DDoubleTensorFromFloat(attr.Floats)
		case "intercepts":
			l.intercepts = tensor.Create1DDoubleTensorFromFloat(attr.Floats)
		default:
			return fmt.Errorf("%s not supported for %s", attr.Name, node.OpType)
		}
	}

	if l.intercepts != nil {
		coeffLen := l.coefficients.Shape[0]
		interLen := l.intercepts.Shape[0]
		if coeffLen%interLen != 0 {
			return fmt.Errorf("coefficient length %d should be divisible by intercepts length %d", coeffLen, interLen)
		}
		l.coefficients.Shape = []int{interLen, coeffLen / interLen}
	}

	l.outputs = make([]int, len(node.Output))
	for i, output := range node.Output {
		l.outputs[i] = k.RegisterWriter(output)
	}
	return nil

}

func (l *LinearRegressor) Compute(k *kernel.Kernel) error {
	data, err := k.Input(l.input)
	if err != nil {
		return err
	}

	input := data.Tensor
	if len(input.Shape) > 2 {
		return fmt.Errorf("linearregressor: invalid shape %v", input.Shape)
	}

	if len(input.Shape) == 1 {
		input.Shape = []int{1, input.Shape[0]}
	}

	if l.intercepts == nil {
		coeffLen := l.coefficients.Shape[0]
		if coeffLen%input.Shape[1] != 0 {
			return fmt.Errorf("coefficient length %d should be divisible by intercepts length %d", coeffLen, input.Shape[1])
		}
		l.coefficients.Shape = []int{coeffLen / input.Shape[1], input.Shape[1]}
	} else if input.Shape[1] != l.coefficients.Shape[1] {
		return fmt.Errorf("input with shape %v cannot be multiplied with coeffiecient of shape %v", input.Shape, l.coefficients.Shape)
	}

	num_classes := l.coefficients.Shape[0]
	num_batches := input.Shape[0]

	scores, err := k.Output(l.outputs[0], []int{num_batches, num_classes}, tensor.Double)
	if err != nil {
		return err
	}

	input.Cast(tensor.Double)

	scores, err = input.Dot(l.coefficients, scores)
	if err != nil {
		return err
	}

	if l.intercepts != nil {
		scores, err = scores.Add(l.intercepts, scores)
		if err != nil {
			return err
		}
	}

	scores.Cast(tensor.Float)
	return nil

}
