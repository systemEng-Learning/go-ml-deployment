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
		 
		val := T(1.0 / (1.0 + math.Exp(-float64(data[i]))))
		if val < 0 {
			data[i] = 1.0 - val
		} else {
			data[i] = val
		}
	}
}