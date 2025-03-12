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
	classlabels_strings [][]byte
	array_like          bool
	output              int
}

func (z *ZipMap) Init(k *kernel.Kernel, node *ir.NodeProto) error {
	input, err := k.RegisterReader(node.Input[0])
	if err != nil {
		return err
	}
	z.input = input
	z.array_like = false
	for _, attr := range node.Attribute {
		switch attr.Name {
		case "classlabels_int64s":
			z.classlabels_int64s = attr.Ints
			array_like := true
			for i := range attr.Ints {
				if int64(i) != attr.Ints[i] {
					array_like = false
				}
			}
			z.array_like = array_like
		case "classlabels_strings":
			z.classlabels_strings = attr.Strings
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
	if z.array_like {
		if data.Readers == 1 {
			k.Put(z.output, input)
		} else {
			output, err := input.Clone()
			if err != nil {
				return err
			}
			k.Put(z.output, output)
		}
		return nil
	}
	if input.DType != tensor.Float {
		return errors.ErrUnsupported
	}
	if z.classlabels_int64s == nil {
		return errors.ErrUnsupported
	}
	output, err := k.Output(z.output, []int{input.Shape[0]}, tensor.IntMap)
	if err != nil {
		return err
	}
	for i := range input.Shape[0] {
		output.IntMap[i] = make(map[int]float32)
		for j := range input.Shape[1] {
			output.IntMap[i][int(z.classlabels_int64s[j])] = input.FloatData[i*input.Shape[1]+j]
		}
	}
	return nil
}
