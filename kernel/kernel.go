package kernel

import (
	"fmt"

	tensors "github.com/systemEng-Learning/go-ml-deployment/tensor"
)

type Kernel struct {
	tensors   []*tensors.Tensor
	tensorMap map[string]int // map of tensor name to index in tensors slice. Only used temporarily during setup
}

func (k *Kernel) Init() {
	k.tensors = make([]*tensors.Tensor, 0)
	k.tensorMap = make(map[string]int)
}

func (k *Kernel) GetTensorIndex(name string) (int, error) {
	index, ok := k.tensorMap[name]
	if !ok {
		return -1, fmt.Errorf("tensor with name %s does not exist", name)
	}
	return index, nil
}

func (k *Kernel) RegisterTensor(name string, tensor *tensors.Tensor) int {
	index, ok := k.tensorMap[name]
	if !ok {
		k.tensors = append(k.tensors, tensor)
		index = len(k.tensors) - 1
		k.tensorMap[name] = index
	}
	return index
}
