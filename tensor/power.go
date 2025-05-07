package tensor

import "math"

func (t *Tensor) Square() {
	length := t.Shape[0]
	if len(t.Shape) == 2 {
		length *= t.Shape[1]
	}
	for i := range length {
		t.FloatData[i] = t.FloatData[i] * t.FloatData[i]
	}
}

func (t *Tensor) Cube() {
	length := t.Shape[0]
	if len(t.Shape) == 2 {
		length *= t.Shape[1]
	}
	for i := range length {
		t.FloatData[i] = t.FloatData[i] * t.FloatData[i] * t.FloatData[i]
	}
}

func (t *Tensor) Power(degree float64) {
	length := t.Shape[0]
	if len(t.Shape) == 2 {
		length *= t.Shape[1]
	}
	for i := range length {
		t.FloatData[i] = float32(math.Pow(float64(t.FloatData[i]), degree))
	}
}
