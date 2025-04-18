package tensor

import (
	"errors"
	"math"
)

// Inverse error function (erf_inv)
func erfInv(x float64) float64 {
	sgn := -1.0
	if x >= 0 {
		sgn = 1.0
	}
	x = (1.0 - x) * (1.0 + x)
	if x == 0 {
		return 0
	}
	log := math.Log(x)
	v := 2.0/(math.Pi*0.147) + 0.5*log
	v2 := 1.0 / 0.147 * log
	v3 := -v + math.Sqrt(v*v-v2)
	return sgn * math.Sqrt(v3)
}

// Compute Probit for a single value
func ComputeProbit(val float64) float64 {
	return 1.41421356 * erfInv(val*2-1)
}

func (t *Tensor) ProbitInPlace() error {
	if len(t.Shape) != 2 || (t.DType != Float && t.DType != Double) {
		return errors.New("unsupported tensor shape or data type")
	}

	switch t.DType {
	case Float:
		Probit(t.FloatData)
	case Double:
		Probit(t.DoubleData)
	}
	return nil
}

// Apply Probit to a slice of data
func Probit[T Float32_64](data []T) {
	for i := range data {
		data[i] = T(ComputeProbit(float64(data[i])))
	}
}
