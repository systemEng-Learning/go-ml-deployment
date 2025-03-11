package tensors

import (
	"errors"
	"math"
)

func (t *Tensor) SoftmaxInPlace() error {
	if len(t.Shape) != 2 || t.DType != Double {
		return errors.ErrUnsupported
	}
	var sum float64
	for i := range t.Shape[0] {
		sum = 0
		for j := range t.Shape[1] {
			t.DoubleData[i*t.Shape[1]+j] = math.Exp(t.DoubleData[i*t.Shape[1]+j])
			sum += t.DoubleData[i*t.Shape[1]+j]
		}

		for j := range t.Shape[1] {
			t.DoubleData[i*t.Shape[1]+j] = t.DoubleData[i*t.Shape[1]+j] / sum
		}
	}
	return nil
}
