package tests

import (
	"testing"

	"github.com/systemEng-Learning/go-ml-deployment/tensor"
)

func runTest[T tensor.Numeric](t *testing.T, input []T, shape []int, output []float32, norm []byte, raiseError bool) {
	sg := Test("Normalizer")
	sg.addAttribute("norm", norm)
	sg.addInput("X", shape, input)
	sg.addOutput("Y", output)
	sg.errorBound = 0.0000001
	err := sg.Execute(t)
	if raiseError && err == nil {
		t.Fatal("expected error")
	}
}

func runTests[T tensor.Numeric](t *testing.T, input []T, shape []int, maxOutput, l1Output, l2Output []float32) {
	runTest(t, input, shape, maxOutput, []byte("MAX"), false)
	runTest(t, input, shape, l1Output, []byte("L1"), false)
	runTest(t, input, shape, l2Output, []byte("L2"), false)
}

func TestNormalizer(t *testing.T) {
	shape := []int{3}
	input := []float32{-1, 0, 1}

	max_output := []float32{-1, 0, 1}
	l1_output := []float32{-0.5, 0, 0.5}
	l2_output := []float32{-0.70710677, 0, 0.70710677}

	runTests(t, input, shape, max_output, l1_output, l2_output)
}
