package ops

import (
	"fmt"

	"github.com/systemEng-Learning/go-ml-deployment/ir"
	"github.com/systemEng-Learning/go-ml-deployment/kernel"
	"github.com/systemEng-Learning/go-ml-deployment/tensor"
)

type DictVectorizer struct {
	input int
	outputs []int
	string_vocabulary [][]byte
	int64_vocabulary []int64
}

func (d *DictVectorizer) Init(k *kernel.Kernel, node *ir.NodeProto) error {
	input, err := k.RegisterReader(node.Input[0])
	if err != nil {
		return err
	}
	d.input = input

	for _, attr := range node.Attribute {
		switch attr.Name {
		case "string_vocabulary":
			d.string_vocabulary = attr.Strings
		case "int64_vocabulary":
			d.int64_vocabulary = attr.Ints
		default:
			return fmt.Errorf("%s not supported for %s", attr.Name, node.OpType)
		}
	}

	d.outputs = make([]int, len(node.Output))

	for i, output := range node.Output {
		d.outputs[i] = k.RegisterWriter(output)
	}

	return nil

}

func (d *DictVectorizer) Compute(k *kernel.Kernel) error {
	data, err := k.Input(d.input)
	if err != nil {
		return err
	}
	input := data.Tensor
}