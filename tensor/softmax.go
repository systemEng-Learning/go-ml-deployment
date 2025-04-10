package tensor

import (
	"errors"
	"math"
)

func (t *Tensor) SoftmaxInPlace() error {
	if len(t.Shape) != 2 || t.DType != Double || t.DType != Float {
		return errors.ErrUnsupported
	}

	switch t.DType {
	case Float:
		SoftMax(t.Shape, t.FloatData)
	case Double:
		SoftMax(t.Shape, t.DoubleData)
	}
	return nil
}

type Float32_64 interface {
	float32 | float64
}

func SoftMax[T Float32_64](shape []int, data []T) {
	for i :=0; i < shape[0]; i++ {
		start := i * shape[1]
		end := start + shape[1]
		row := data[start:end]
		max := row[0]
		for _, val := range row {
			if val > max {
				max = val
			}
		}
		var sum T
		for j := range row {
			row[j] = T(math.Exp(float64(row[j] - max)))
			sum += row[j]
		}
		for j := range row {
			if sum != 0 {
				row[j] /= sum
			}
		}
	}
}