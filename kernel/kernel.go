package kernel

import (
	"fmt"

	tensors "github.com/systemEng-Learning/go-ml-deployment/tensor"
)

/*
 * Some operations, such as `cast` and `zipmap`, do not modify their input tensors sometimes.
 * Instead of cloning the input tensor unnecessarily, we can directly reuse the same tensor
 * in the output by passing its pointer.
 *
 * However, since pointers can be modified by other operations, we must ensure that it is
 * safe to reuse the tensor. To do this, we use a `Readers` attribute:
 *
 * - If `Readers == 1`, it is safe to transfer the tensor pointer.
 * - Otherwise, cloning might be necessary to prevent unintended modifications.
 */
type Data struct {
	Readers int
	Tensor  *tensors.Tensor
}

/*
 * The Kernel struct manages tensors and allows operations to fetch them efficiently.
 * - Tensors are stored in a slice (`tensors`), and operations access them using an index.
 * - During setup, a temporary `tensorMap` is used to map tensor names to their indices.
 * - Once setup is complete, operations can retrieve tensors directly using the index.
 */
type Kernel struct {
	tensors   []Data
	tensorMap map[string]int // map of tensor name to index in tensors slice. Only used temporarily during setup
}

// Initialize kernel and its members
func (k *Kernel) Init() {
	k.tensors = make([]Data, 0)
	k.tensorMap = make(map[string]int)
}

// Register as a reader for a tensor using the name as key. This increments
// the number of reader as a side-effect. Returns the position of the tensor in the kernel.
func (k *Kernel) RegisterReader(name string) (int, error) {
	index, ok := k.tensorMap[name]
	if !ok {
		return -1, fmt.Errorf("tensor with name %s does not exist", name)
	}
	k.tensors[index].Readers++
	return index, nil
}

// Register as a writer for a tensor using the name as key. Returns the position of the
// tensor in the kernel
func (k *Kernel) RegisterWriter(name string) int {
	index, ok := k.tensorMap[name]
	if !ok {
		k.tensors = append(k.tensors, Data{})
		index = len(k.tensors) - 1
		k.tensorMap[name] = index
	}
	return index
}

// Get a tensor from the kernel using its index
func (k *Kernel) Input(index int) (Data, error) {
	var d Data
	if index >= len(k.tensors) {
		return d, fmt.Errorf("tensor with index %d does not exist", index)
	}
	return k.tensors[index], nil
}

/*
* Output retrieves or initializes a tensor at the given index in the Kernel.
*
  - - If the index is out of range, an error is returned.
  - - If the tensor at the specified index is `nil`, a new tensor is created with the given shape and data type.
  - - If a tensor already exists:
  - - If the data type has changed and it's not related i.e float & double, int32 & int64,
    the tensor is cleared and updated.
  - - If the existing tensor’s capacity is insufficient, memory is reallocated.

*
* This function ensures that a valid tensor is always returned for the given index.
*/
func (k *Kernel) Output(index int, shape []int, dtype tensors.DataType) (*tensors.Tensor, error) {
	if index >= len(k.tensors) {
		return nil, fmt.Errorf("tensor with index %d does not exist", index)
	}
	d := k.tensors[index]
	t := d.Tensor

	if t == nil {
		t = &tensors.Tensor{
			Shape: shape,
			DType: dtype,
		}
		t.Alloc()
		k.tensors[index].Tensor = t
	} else {
		count := shape[0]
		if len(shape) > 1 {
			count *= shape[1]
		}
		capacity := 0
		if dtype == t.DType || (t.DType == tensors.Double && dtype == tensors.Float) ||
			(t.DType == tensors.Float && dtype == tensors.Double) || (t.DType == tensors.Int64 && dtype == tensors.Int32) ||
			(t.DType == tensors.Int32 && dtype == tensors.Int64) {
			capacity = t.Capacity()
		} else {
			t.Clear()
		}
		t.DType = dtype
		t.Shape = shape
		if capacity < count {
			t.Alloc()
		}
	}
	return t, nil
}

// Inserts a tensor at a given index in the kernel
func (k *Kernel) Put(index int, tensor *tensors.Tensor) error {
	if index >= len(k.tensors) {
		return fmt.Errorf("tensor with index %d does not exist", index)
	}
	k.tensors[index].Tensor = tensor
	return nil
}

// Retrieves a tensor at a given index in the kernel
func (k *Kernel) Get(index int) *tensors.Tensor {
	if index >= len(k.tensors) {
		return nil
	}
	return k.tensors[index].Tensor
}
