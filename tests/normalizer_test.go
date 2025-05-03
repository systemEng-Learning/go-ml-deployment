package tests

import (
	"testing"
)

type normalizerInput interface {
	float32 | float64 | int32
}

type outputType interface {
	float32 | []float32
}

func runTest[T normalizerInput, U outputType](t *testing.T, input []T, shape []int, output []U, norm []byte, raiseError bool) {
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

func runTests[T normalizerInput, U outputType](t *testing.T, input []T, shape []int, maxOutput, l1Output, l2Output []U) {
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

func TestNormalizer2DFloats(t *testing.T) {
	shape := []int{2, 3}
	input := []float32{-1.0856306, 0.99734545, 0.2829785, -1.50629471, -0.57860025, 1.65143654}

	max_output := [][]float32{{-1.0885202, 1, 0.2837317}, {-0.91211176, -0.35036176, 1}}
	l1_output := [][]float32{{-0.45885524, 0.42154038, 0.11960436}, {-0.40314806, -0.15485784, 0.44199413}}
	l2_output := [][]float32{{-0.7232126, 0.6643998, 0.18851127}, {-0.65239084, -0.25059736, 0.7152532}}
	runTests(t, input, shape, max_output, l1_output, l2_output)
}

func TestNormalizer2DDoubles(t *testing.T) {
	shape := []int{2, 3}
	input := []float64{-1.0856306, 0.99734545, 0.2829785, -1.50629471, -0.57860025, 1.65143654}

	max_output := [][]float32{{-1.0885202, 1, 0.2837317}, {-0.91211176, -0.35036176, 1}}
	l1_output := [][]float32{{-0.45885524, 0.42154038, 0.11960436}, {-0.40314806, -0.15485784, 0.44199413}}
	l2_output := [][]float32{{-0.7232126, 0.6643998, 0.18851127}, {-0.65239084, -0.25059736, 0.7152532}}
	runTests(t, input, shape, max_output, l1_output, l2_output)
}

func TestNormalizer2DInt32s(t *testing.T) {
	shape := []int{3, 2}
	input := []int32{-242, -42, 126, -86, -67, -9}
	max_output := [][]float32{{5.7619047, 1}, {1, -0.6825397}, {7.4444447, 1}}
	l1_output := [][]float32{{-0.85211265, -0.14788732}, {0.5943396, -0.4056604}, {-0.8815789, -0.11842106}}
	l2_output := [][]float32{{-0.98527145, -0.17099753}, {0.82594985, -0.56374353}, {-0.9910982, -0.13313259}}
	runTests(t, input, shape, max_output, l1_output, l2_output)
}

func TestNormalizerInvalidNorm(t *testing.T) {
	shape := []int{3}
	input := []float32{-1, 0, 1}

	output := []float32{-1, 0, 1}

	runTest(t, input, shape, output, []byte("InvalidNorm"), true)
}
