package ops

import (
	"errors"
	"fmt"

	"github.com/systemEng-Learning/go-ml-deployment/ir"
	"github.com/systemEng-Learning/go-ml-deployment/kernel"
	"github.com/systemEng-Learning/go-ml-deployment/tensor"
)

type ZipMap struct {
	input               int
	classlabels_int64s  []int64
	classlabels_strings []string
	use_strings         bool
	output              int
}

func (z *ZipMap) Init(k *kernel.Kernel, node *ir.NodeProto) error {
	input, err := k.RegisterReader(node.Input[0])
	if err != nil {
		return err
	}
	z.input = input
	z.use_strings = false
	for _, attr := range node.Attribute {
		switch attr.Name {
		case "classlabels_int64s":
			z.classlabels_int64s = attr.Ints
		case "classlabels_strings":
			z.classlabels_strings = make([]string, len(attr.Strings))
			for i := range attr.Strings {
				z.classlabels_strings[i] = string(attr.Strings[i])
			}
			z.use_strings = true
		default:
			return fmt.Errorf("%s not supported for %s", attr.Name, node.OpType)
		}
	}
	z.output = k.RegisterWriter(node.Output[0])
	return nil
}

func (z *ZipMap) Compute(k *kernel.Kernel) error {
	data, err := k.Input(z.input)
	if err != nil {
		return err
	}
	input := data.Tensor
	if input.DType != tensor.Float {
		return errors.ErrUnsupported
	}
	rows := input.Shape[0]
	cols := rows
	if len(input.Shape) == 1 {
		rows = 1
	} else {
		cols = input.Shape[1]
	}
	if z.use_strings {
		if cols != len(z.classlabels_strings) {
			return fmt.Errorf("zipmap: input features per batch %d != number of classlabels %d", cols, len(z.classlabels_strings))
		}
		output, err := k.Output(z.output, []int{rows}, tensor.StringMap)
		if err != nil {
			return err
		}
		for i := range rows {
			output.StringMap[i] = make(map[string]float32)
			for j := range cols {
				output.StringMap[i][z.classlabels_strings[j]] = input.FloatData[i*cols+j]
			}
		}
	} else {
		if cols != len(z.classlabels_int64s) {
			return fmt.Errorf("zipmap: input features per batch %d != number of classlabels %d", cols, len(z.classlabels_int64s))
		}
		output, err := k.Output(z.output, []int{rows}, tensor.IntMap)
		if err != nil {
			return err
		}
		for i := range rows {
			output.IntMap[i] = make(map[int64]float32)
			for j := range cols {
				output.IntMap[i][z.classlabels_int64s[j]] = input.FloatData[i*cols+j]
			}
		}
	}
	return nil
}
