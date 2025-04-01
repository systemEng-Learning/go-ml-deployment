package ops

import (
	"fmt"

	"github.com/systemEng-Learning/go-ml-deployment/ir"
	"github.com/systemEng-Learning/go-ml-deployment/kernel"
	"github.com/systemEng-Learning/go-ml-deployment/tensor"
)

type TreeEnsembleClassifier struct {
	tree *TreeEnsemble
	input int
	outputs []int
}

func (t *TreeEnsembleClassifier) Init(k *kernel.Kernel, node *ir.NodeProto) error {
	input, err := k.RegisterReader(node.Input[0])
	if err != nil {
		return err
	}
	t.input = input

	t.tree = &TreeEnsemble{}
	err = t.tree.Init(node)
	if err != nil {
		return err
	}
}