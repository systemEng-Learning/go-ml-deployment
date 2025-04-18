package tensor

import (
	"errors"
	"math"
)

func (t *Tensor) LogisticInPlace() error {
	if len(t.Shape) != 2 || (t.DType != Float && t.DType != Double) {
		return errors.New("unsupported tensor shape or data type")
	}

	switch t.DType {
	case Float:
		Logistic(t.FloatData)
	case Double:
		Logistic(t.DoubleData)
	}
	return nil
}

func Logistic[T Float32_64](data []T) {
	for i := range data {
		data[i] = T(ComputeLogistic(float64(data[i])))
	}
}

func ComputeLogistic(val float64) float64 {
	v := (1.0 / (1.0 + math.Exp(-val)))
	if v < 0 {
		return 1.0 - v
	}
	return v
}
