package ops

import (
	"errors"
	"fmt"

	"github.com/systemEng-Learning/go-ml-deployment/ir"
	"github.com/systemEng-Learning/go-ml-deployment/kernel"
	"github.com/systemEng-Learning/go-ml-deployment/tensor"
)

type LinearClassifier struct {
	input             int
	classlabel        []int64
	classlabel_string [][]byte
	coefficients      *tensor.Tensor
	intercepts        *tensor.Tensor
	multiclass        bool
	post_transform    string
	outputs           []int
}

func (l *LinearClassifier) Init(k *kernel.Kernel, node *ir.NodeProto) error {
	input, err := k.RegisterReader(node.Input[0])
	if err != nil {
		return err
	}
	l.input = input
	l.multiclass = false
	l.post_transform = "NONE"

	for _, attr := range node.Attribute {
		switch attr.Name {
		case "classlabels_ints":
			l.classlabel = attr.Ints
		case "classlabels_strings":
			l.classlabel_string = attr.Strings
		case "coefficients":
			l.coefficients = tensor.Create1DDoubleTensorFromFloat(attr.Floats)
		case "intercepts":
			l.intercepts = tensor.Create1DDoubleTensorFromFloat(attr.Floats)
		case "multi_class":
			if attr.I > 0 {
				l.multiclass = true
			}
		case "post_transform":
			l.post_transform = string(attr.S)
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

func (l *LinearClassifier) Compute(k *kernel.Kernel) error {
	data, err := k.Input(l.input)
	if err != nil {
		return err
	}
	input := data.Tensor
	if len(input.Shape) > 2 {
		return fmt.Errorf("linearclassifier: invalid shape %v", input.Shape)
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

	var labels *tensor.Tensor
	if l.classlabel != nil {
		labels, err = k.Output(l.outputs[0], []int{num_batches}, tensor.Int64)
		if err != nil {
			return err
		}
	} else {
		return errors.ErrUnsupported
	}
	scores, err := k.Output(l.outputs[1], []int{num_batches, num_classes}, tensor.Double)
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

	if l.classlabel != nil {
		for d := range scores.Shape[0] {
			max_class := 0
			row := d * scores.Shape[1]
			max_weight := scores.DoubleData[row]
			for i := 1; i < scores.Shape[1]; i++ {
				if scores.DoubleData[row+i] > max_weight {
					max_class = i
					max_weight = scores.DoubleData[row+i]
				}
			}
			labels.Int64Data[d] = l.classlabel[max_class]
		}
	}

	if l.post_transform == "SOFTMAX" {
		scores.SoftmaxInPlace()
	} else if l.post_transform != "NONE" {
		return errors.ErrUnsupported
	}
	scores.Cast(tensor.Float)
	return nil
}
