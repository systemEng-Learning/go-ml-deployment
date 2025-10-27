package tensor

import "fmt"

func (t *Tensor) SizeHelper(start, end int) int64 {
	var size int64 = 1
	for i := start; i < end; i++ {
		if t.Shape[i] < 0 {
			return -1
		}
		size *= int64(t.Shape[i])
	}
	return size
}

func (t *Tensor) Size() int64 {
	return t.SizeHelper(0, len(t.Shape))
}

func (t *Tensor) SizeToDimension(dimension int) (int64, error) {
	num_dims := len(t.Shape)
	if dimension > num_dims {
		return 0, fmt.Errorf("sizetodimension: invalid dimension of %d. Tensor has %d dimensions", dimension, num_dims)
	}
	return t.SizeHelper(0, dimension), nil
}

func (t *Tensor) SizeFromDimension(dimension int) (int64, error) {
	num_dims := len(t.Shape)
	if dimension > num_dims {
		return 0, fmt.Errorf("sizefromdimension: invalid dimension of %d. Tensor has %d dimensions", dimension, num_dims)
	}
	return t.SizeHelper(dimension, num_dims), nil
}
