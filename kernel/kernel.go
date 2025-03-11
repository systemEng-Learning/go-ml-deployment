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

func (k *Kernel) RegisterTensor(name string) int {
	index, ok := k.tensorMap[name]
	if !ok {
		k.tensors = append(k.tensors, nil)
		index = len(k.tensors) - 1
		k.tensorMap[name] = index
	}
	return index
}

func (k *Kernel) Input(index int) (*tensors.Tensor, error) {
	if index >= len(k.tensors) {
		return nil, fmt.Errorf("tensor with index %d does not exist", index)
	}
	return k.tensors[index], nil
}

func (k *Kernel) Output(index int, shape []int, dtype tensors.DataType) (*tensors.Tensor, error) {
	if index >= len(k.tensors) {
		return nil, fmt.Errorf("tensor with index %d does not exist", index)
	}
	t := k.tensors[index]
	if t == nil {
		t = &tensors.Tensor{
			Shape: shape,
			DType: dtype,
		}
		t.Alloc()
		k.tensors[index] = t
	} else {
		count := shape[0]
		if len(shape) > 1 {
			count *= shape[1]
		}
		capacity := 0
		if dtype != t.DType {
			t.Clear()
			t.DType = dtype
		} else {
			capacity = t.Capacity()
		}
		t.Shape = shape
		if capacity < count {
			t.Alloc()
		}
	}
	return t, nil
}

func (k *Kernel) Put(index int, tensor *tensors.Tensor) error {
	if index >= len(k.tensors) {
		return fmt.Errorf("tensor with index %d does not exist", index)
	}
	k.tensors[index] = tensor
	return nil
}

func (k *Kernel) Get(index int) *tensors.Tensor {
	if index >= len(k.tensors) {
		return nil
	}
	return k.tensors[index]
}
