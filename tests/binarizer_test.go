package tests

import "testing"

type BinarizerInput interface {
	float32 | float64 | int32 | int64
}

func runBinarizerTest[T BinarizerInput](t *testing.T, input []T, shape []int, threshold float32, expected []T) {
	sg := Test("Binarizer")
	sg.addAttribute("threshold", threshold)
	sg.addInput("X", shape, input)
	sg.addOutput("Y", expected)
	sg.errorBound = 0.0000001
	sg.Execute(t)
}

func TestBinarizerFloat32(t *testing.T) {
	input := []float32{-1.5, -0.5, 0, 0.5, 1.5, 2.5}
	expected := []float32{0, 0, 0, 1, 1, 1}
	runBinarizerTest(t, input, []int{6}, 0.0, expected)
}

func TestBinarizerFloat32WithThreshold(t *testing.T) {
	input := []float32{-1.5, -0.5, 0, 0.5, 1.5, 2.5}
	expected := []float32{0, 0, 0, 0, 1, 1}
	runBinarizerTest(t, input, []int{6}, 1.0, expected)
}

func TestBinarizerFloat64(t *testing.T) {
	input := []float64{-1.5, -0.5, 0, 0.5, 1.5, 2.5}
	expected := []float64{0, 0, 0, 1, 1, 1}
	runBinarizerTest(t, input, []int{6}, 0.0, expected)
}

func TestBinarizerFloat64WithThreshold(t *testing.T) {
	input := []float64{-1.5, -0.5, 0, 0.5, 1.5, 2.5}
	expected := []float64{0, 0, 0, 0, 1, 1}
	runBinarizerTest(t, input, []int{6}, 1.0, expected)
}

func TestBinarizerInt32(t *testing.T) {
	input := []int32{-2, -1, 0, 1, 2, 3}
	expected := []int32{0, 0, 0, 1, 1, 1}
	runBinarizerTest(t, input, []int{6}, 0.0, expected)
}

func TestBinarizerInt32WithThreshold(t *testing.T) {
	input := []int32{-2, -1, 0, 1, 2, 3}
	expected := []int32{0, 0, 0, 0, 1, 1}
	runBinarizerTest(t, input, []int{6}, 1.5, expected)
}

func TestBinarizerInt64(t *testing.T) {
	input := []int64{-2, -1, 0, 1, 2, 3}
	expected := []int64{0, 0, 0, 1, 1, 1}
	runBinarizerTest(t, input, []int{6}, 0.0, expected)
}

func TestBinarizerInt64WithThreshold(t *testing.T) {
	input := []int64{-2, -1, 0, 1, 2, 3}
	expected := []int64{0, 0, 0, 0, 1, 1}
	runBinarizerTest(t, input, []int{6}, 1.5, expected)
}

func TestBinarizer2DFloat32(t *testing.T) {
	input := [][]float32{{-1.5, 0.5}, {1.5, -0.5}, {2.5, 0}}
	expected := [][]float32{{0, 1}, {1, 0}, {1, 0}}
	sg := Test("Binarizer")
	sg.addAttribute("threshold", float32(0.0))
	sg.addInput("X", []int{3, 2}, input)
	sg.addOutput("Y", expected)
	sg.errorBound = 0.0000001
	sg.Execute(t)
}

func TestBinarizer2DInt32(t *testing.T) {
	input := [][]int32{{-2, 1}, {2, -1}, {3, 0}}
	expected := [][]int32{{0, 1}, {1, 0}, {1, 0}}
	sg := Test("Binarizer")
	sg.addAttribute("threshold", float32(0.0))
	sg.addInput("X", []int{3, 2}, input)
	sg.addOutput("Y", expected)
	sg.errorBound = 0.0000001
	sg.Execute(t)
}

func TestBinarizerNegativeThreshold(t *testing.T) {
	input := []float32{-2.0, -1.0, -0.5, 0, 0.5, 1.0}
	expected := []float32{0, 1, 1, 1, 1, 1}
	runBinarizerTest(t, input, []int{6}, -1.5, expected)
}

func TestBinarizerHighThreshold(t *testing.T) {
	input := []float32{-1.0, 0, 1.0, 2.0, 5.0, 10.0}
	expected := []float32{0, 0, 0, 0, 0, 1}
	runBinarizerTest(t, input, []int{6}, 7.5, expected)
}

func TestBinarizerWithAttributeNameT(t *testing.T) {
	sg := Test("Binarizer")
	sg.addAttribute("t", float32(0.5))
	input := []float32{0, 0.3, 0.7, 1.0}
	expected := []float32{0, 0, 1, 1}
	sg.addInput("X", []int{4}, input)
	sg.addOutput("Y", expected)
	sg.errorBound = 0.0000001
	sg.Execute(t)
}