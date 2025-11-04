package ops

import (
	"fmt"
	"slices"

	"github.com/systemEng-Learning/go-ml-deployment/ir"
	"github.com/systemEng-Learning/go-ml-deployment/kernel"
	"github.com/systemEng-Learning/go-ml-deployment/tensor"
)

type OneHotEncoder struct {
	input          int
	output         int
	cat_int        map[int]int64
	cat_string     map[string]int64
	num_categories int64
	zeros          bool
}

func (o *OneHotEncoder) Init(k *kernel.Kernel, node *ir.NodeProto) error {
	input, err := k.RegisterReader(node.Input[0])
	if err != nil {
		return err
	}
	o.input = input
	o.zeros = true
	for _, attr := range node.Attribute {
		switch attr.Name {
		case "cats_int64s":
			o.cat_int = make(map[int]int64)
			o.num_categories = int64(len(attr.Ints))
			for i, v := range attr.Ints {
				o.cat_int[int(v)] = int64(i)
			}
		case "cats_strings":
			o.cat_string = make(map[string]int64)
			o.num_categories = int64(len(attr.Strings))
			for i, v := range attr.Strings {
				o.cat_string[string(v)] = int64(i)
			}
		case "zeros":
			if attr.I == 0 {
				o.zeros = false
			}
		default:
			return fmt.Errorf("%s not supported for %s", attr.Name, node.OpType)
		}
	}
	if o.num_categories == 0 {
		return fmt.Errorf("no category was defined for %s", node.OpType)
	}
	o.output = k.RegisterWriter(node.Output[0])
	return nil
}

func (o *OneHotEncoder) Compute(k *kernel.Kernel) error {
	data, err := k.Input(o.input)
	input := data.Tensor
	if err != nil {
		return err
	}
	output_shape := slices.Clone(input.Shape)
	output_shape = append(output_shape, int(o.num_categories))
	output, err := k.Output(o.output, output_shape, tensor.Float)
	if err != nil {
		return err
	}
	clear(output.FloatData[:output.Size()])

	switch input.DType {
	case tensor.Int32:
		err = ohe(input.Int32Data, input.Size(), output.FloatData, o.cat_int, o.num_categories, o.zeros)
	case tensor.Int64:
		err = ohe(input.Int64Data, input.Size(), output.FloatData, o.cat_int, o.num_categories, o.zeros)
	case tensor.Float:
		err = ohe(input.FloatData, input.Size(), output.FloatData, o.cat_int, o.num_categories, o.zeros)
	case tensor.Double:
		err = ohe(input.DoubleData, input.Size(), output.FloatData, o.cat_int, o.num_categories, o.zeros)
	case tensor.String:
		err = oheString(input.StringData, input.Size(), output.FloatData, o.cat_string, o.num_categories, o.zeros)
	default:
		return fmt.Errorf("onehotencoder: invalid tensor data type")
	}

	return err
}

func ohe[T tensor.Numeric](input []T, input_size int64, output []float32, category map[int]int64, num_categories int64, zeros bool) error {
	for i := range input_size {
		idx, ok := category[int(input[i])]
		if ok {
			output[i*num_categories+idx] = 1
		} else if !zeros {
			return fmt.Errorf("onehotencoder: Unknown category and zeros = 0")
		}
	}
	return nil
}

func oheString(input [][]byte, input_size int64, output []float32, category map[string]int64, num_categories int64, zeros bool) error {
	for i := range input_size {
		idx, ok := category[string(input[i])]
		if ok {
			output[i*num_categories+idx] = 1
		} else if !zeros {
			return fmt.Errorf("onehotencoder: Unknown category and zeros = 0")
		}
	}
	return nil
}
