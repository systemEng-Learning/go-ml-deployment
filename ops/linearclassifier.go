package ops

import (
	"fmt"

	"github.com/systemEng-Learning/go-ml-deployment/ir"
	"github.com/systemEng-Learning/go-ml-deployment/kernel"
	"github.com/systemEng-Learning/go-ml-deployment/tensor"
)

type LinearClassifier struct {
	input             int
	num_targets       int
	using_strings     bool
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
	using_strings := false

	for _, attr := range node.Attribute {
		switch attr.Name {
		case "classlabels_ints":
			l.classlabel = attr.Ints
		case "classlabels_strings":
			l.classlabel_string = attr.Strings
			using_strings = true
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
	l.using_strings = using_strings

	if l.intercepts != nil {
		coeffLen := l.coefficients.Shape[0]
		interLen := l.intercepts.Shape[0]
		if coeffLen%interLen != 0 {
			return fmt.Errorf("coefficient length %d should be divisible by intercepts length %d", coeffLen, interLen)
		}
		l.coefficients.Shape = []int{interLen, coeffLen / interLen}
		l.num_targets = interLen
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
	num_targets := l.num_targets
	if l.intercepts == nil {
		coeffLen := l.coefficients.Shape[0]
		if coeffLen%input.Shape[1] != 0 {
			return fmt.Errorf("coefficient length %d should be divisible by intercepts length %d", coeffLen, input.Shape[1])
		}
		num_targets = coeffLen / input.Shape[1]
		l.coefficients.Shape = []int{num_targets, input.Shape[1]}
	} else if input.Shape[1] != l.coefficients.Shape[1] {
		return fmt.Errorf("input with shape %v cannot be multiplied with coeffiecient of shape %v", input.Shape, l.coefficients.Shape)
	}

	num_batches := input.Shape[0]

	var output_dtype tensor.DataType
	if l.using_strings {
		output_dtype = tensor.String
	} else {
		output_dtype = tensor.Int64
	}
	labels, err := k.Output(l.outputs[0], []int{num_batches}, output_dtype)
	if err != nil {
		return err
	}

	output_classes := num_targets
	add_second_class := false
	if num_targets == 1 && ((l.using_strings && len(l.classlabel_string) == 2) || (!l.using_strings && len(l.classlabel) == 2)) {
		output_classes = 2
		add_second_class = true
	}
	scores, err := k.Output(l.outputs[1], []int{num_batches, output_classes}, tensor.Double)
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

	if num_targets == 1 {
		if l.using_strings {
			use_class_labels := len(l.classlabel_string) == 2
			positive_label := []byte("1")
			negative_label := []byte("0")
			if use_class_labels {
				positive_label = l.classlabel_string[1]
				negative_label = l.classlabel_string[0]
			}
			for d := range scores.Shape[0] {
				if scores.DoubleData[d] > 0 {
					labels.StringData[d] = positive_label
				} else {
					labels.StringData[d] = negative_label
				}
			}
		} else {
			use_class_labels := len(l.classlabel) == 2
			positive_label := int64(1)
			negative_label := int64(0)
			if use_class_labels {
				positive_label = l.classlabel[1]
				negative_label = l.classlabel[0]
			}
			for d := range scores.Shape[0] {
				if scores.DoubleData[d] > 0 {
					labels.Int64Data[d] = positive_label
				} else {
					labels.Int64Data[d] = negative_label
				}
			}
		}
	} else {
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
			if l.using_strings {
				labels.StringData[d] = l.classlabel_string[max_class]
			} else {
				labels.Int64Data[d] = l.classlabel[max_class]
			}
		}
	}
	if l.post_transform != "NONE" || add_second_class {
		to_add := -1
		if add_second_class {
			to_add = 1
		}
		update_scores(scores.DoubleData, []int{num_batches, num_targets}, l.post_transform, to_add, false)
	}
	scores.Cast(tensor.Float)
	return nil
}
