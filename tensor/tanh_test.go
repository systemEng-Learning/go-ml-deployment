package tensor

import (
	"math"
	"testing"
)

func TestTanh(t *testing.T) {
	a := mustTensor(CreateEmptyTensor([]int{2, 3}, Float), []float32{1, 2, 3, 4, 5, 6})
	expected := []float32{float32(math.Tanh(1)), float32(math.Tanh(2)), float32(math.Tanh(3)),
		float32(math.Tanh(4)), float32(math.Tanh(5)), float32(math.Tanh(6))}

	a.Tanh(a)
	if !matricesClose(a.FloatData, expected, 1e-9) {
		t.Errorf("expected %v, got %v", expected, a.FloatData)
	}
}

func matricesClose(a, b []float32, tol float64) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if math.Abs(float64(a[i]-b[i])) > tol {
			return false
		}
	}
	return true
}
