package tests

import (
	"testing"
)

type scalerInput interface {
	float32 | float64 | int32 | int64
}

func scalerTest[T scalerInput](t *testing.T) {
	sg := Test("Scaler")
	scale := []float32{3, -4, 3.0}
	offset := []float32{4.8, -0.5, 77.0}
	sg.addAttribute("scale", scale)
	sg.addAttribute("offset", offset)

	input := []T{1, -2, 3, 4, 5, -6}
	shape := []int{2, 3}
	sg.addInput("X", shape, input)

	output := make([][]float32, shape[0])
	for i := range output {
		output[i] = make([]float32, shape[1])
		for j := range shape[1] {
			output[i][j] = (float32(input[i*shape[1]+j]) - offset[j%shape[1]]) * scale[j%shape[1]]
		}
	}
	sg.addOutput("Y", output)
	sg.errorBound = 0.00001
	err := sg.Execute(t)

	if err != nil {
		t.Fatalf("error shouldn't exist: %v", err)
	}
}

func TestScaler(t *testing.T) {
	scalerTest[float32](t)
	scalerTest[float64](t)
	scalerTest[int32](t)
	scalerTest[int64](t)
}
