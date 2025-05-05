package tensor

import "math"

func (t *Tensor) Tanh(out *Tensor) {
	length := t.Shape[0]
	if len(t.Shape) == 2 {
		length *= t.Shape[1]
	}
	for i := range length {
		out.FloatData[i] = float32(math.Tanh(float64(out.FloatData[i])))
	}
}
