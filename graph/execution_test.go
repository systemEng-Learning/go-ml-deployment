package graph

import (
	"testing"

	"github.com/systemEng-Learning/go-ml-deployment/kernel"
	"github.com/systemEng-Learning/go-ml-deployment/tensor"
)

func TestExecute_CorrectInput(t *testing.T) {
	g := &Graph{
		shapes: [][]int{{3}},
		dtypes: []tensor.DataType{tensor.Float},
		kernel: &kernel.Kernel{},
	}
	g.kernel.Init()
	g.inputs = []int{g.kernel.RegisterWriter("input1")}

	input := []any{[]float32{1.0, 2.0, 3.0}}
	err := g.Execute(input)
	if err != nil {
		t.Errorf("Expected no error, got: %v", err)
	}
}

func TestExecute_StaticValue(t *testing.T) {
	g := &Graph{
		shapes: [][]int{{-1, 1}},
		dtypes: []tensor.DataType{tensor.Float},
		kernel: &kernel.Kernel{},
	}
	g.kernel.Init()
	g.inputs = []int{g.kernel.RegisterWriter("input1")}

	staticValue := float32(42.0)

	err := g.Execute([]any{staticValue})
	if err != nil {
		t.Errorf("Expected no error when processing static value, got: %v", err)
	}
}

func TestExecute_Int64Input(t *testing.T) {
	g := &Graph{
		shapes: [][]int{{3}},
		dtypes: []tensor.DataType{tensor.Int64},
		kernel: &kernel.Kernel{},
	}
	g.kernel.Init()
	g.inputs = []int{g.kernel.RegisterWriter("input1")}

	input := []any{[]int64{1, 2, 3}}
	err := g.Execute(input)
	if err != nil {
		t.Errorf("Expected no error, got: %v", err)
	}
}

func TestExecute_DoubleInput(t *testing.T) {
	g := &Graph{
		shapes: [][]int{{3}},
		dtypes: []tensor.DataType{tensor.Double},
		kernel: &kernel.Kernel{},
	}
	g.kernel.Init()
	g.inputs = []int{g.kernel.RegisterWriter("input1")}

	input := []any{[]float64{1.0, 2.0, 3.0}}
	err := g.Execute(input)
	if err != nil {
		t.Errorf("Expected no error, got: %v", err)
	}
}

func TestExecute_MultipleInputs(t *testing.T) {
	g := &Graph{
		shapes: [][]int{{3}, {3}},
		dtypes: []tensor.DataType{tensor.Float, tensor.Int32},
		kernel: &kernel.Kernel{},
	}
	g.kernel.Init()
	g.inputs = []int{g.kernel.RegisterWriter("input1"), g.kernel.RegisterWriter("input2")}

	input := []any{[]float32{1.0, 2.0, 3.0}, []int32{4, 5, 6}}
	err := g.Execute(input)
	if err != nil {
		t.Errorf("Expected no error for multiple inputs, got: %v", err)
	}
}

func TestExecute_CastInt32ToInt64(t *testing.T) {
	g := &Graph{
		shapes: [][]int{{3}},
		dtypes: []tensor.DataType{tensor.Int64},
		kernel: &kernel.Kernel{},
	}
	g.kernel.Init()
	g.inputs = []int{g.kernel.RegisterWriter("input1")}

	input := []any{[]int32{1, 2, 3}}
	err := g.Execute(input)
	if err != nil {
		t.Errorf("Expected no error when casting int32 to int64, got: %v", err)
	}
}

func TestExecute_CastFloatToDouble(t *testing.T) {
	g := &Graph{
		shapes: [][]int{{3}},
		dtypes: []tensor.DataType{tensor.Double},
		kernel: &kernel.Kernel{},
	}
	g.kernel.Init()
	g.inputs = []int{g.kernel.RegisterWriter("input1")}

	input := []any{[]float32{1.0, 2.0, 3.0}}
	err := g.Execute(input)
	if err != nil {
		t.Errorf("Expected no error when casting float to double, got: %v", err)
	}
}

func TestExecute_CastIntToDouble(t *testing.T) {
	g := &Graph{
		shapes: [][]int{{3}},
		dtypes: []tensor.DataType{tensor.Double},
		kernel: &kernel.Kernel{},
	}
	g.kernel.Init()
	g.inputs = []int{g.kernel.RegisterWriter("input1")}

	input := []any{[]int{1, 2, 3}}
	err := g.Execute(input)
	if err != nil {
		t.Errorf("Expected no error when casting int to double, got: %v", err)
	}
}

func TestExecute_CastDoubleToInt32(t *testing.T) {
	g := &Graph{
		shapes: [][]int{{3}},
		dtypes: []tensor.DataType{tensor.Int32},
		kernel: &kernel.Kernel{},
	}
	g.kernel.Init()
	g.inputs = []int{g.kernel.RegisterWriter("input1")}

	input := []any{[]float64{1.9, 2.5, 3.1}}
	err := g.Execute(input)
	if err != nil {
		t.Errorf("Expected no error when casting float to int32, got: %v", err)
	}
}

func TestExecute_ReshapeInput(t *testing.T) {
	g := &Graph{
		shapes: [][]int{{-1, 3}},
		dtypes: []tensor.DataType{tensor.Float},
		kernel: &kernel.Kernel{},
	}
	g.kernel.Init()
	g.inputs = []int{g.kernel.RegisterWriter("input1")}

	input := []any{[]float32{1.0, 2.0, 3.0, 4.0, 5.0, 6.0}}
	err := g.Execute(input)
	if err != nil {
		t.Errorf("Expected no error when reshaping input, got: %v", err)
	}
}

func TestExecute_EmptyInput(t *testing.T) {
	g := &Graph{
		shapes: [][]int{{3}},
		dtypes: []tensor.DataType{tensor.Float},
		kernel: &kernel.Kernel{},
	}
	g.kernel.Init()
	g.inputs = []int{g.kernel.RegisterWriter("input1")}

	input := []any{} // Empty input
	err := g.Execute(input)
	if err == nil {
		t.Errorf("Expected an error due to input count mismatch, but got none")
	}
}

func TestExecute_InputCountMismatch_TooFew(t *testing.T) {
	g := &Graph{
		shapes: [][]int{{3}, {3}},
		dtypes: []tensor.DataType{tensor.Float, tensor.Float},
		kernel: &kernel.Kernel{},
	}
	g.kernel.Init()
	g.inputs = []int{g.kernel.RegisterWriter("input1"), g.kernel.RegisterWriter("input2")}

	input := []any{[]float32{1.0, 2.0, 3.0}} // Only one input provided
	err := g.Execute(input)
	if err == nil {
		t.Errorf("Expected an error due to too few inputs, but got none")
	}
}

func TestExecute_InputCountMismatch_TooMany(t *testing.T) {
	g := &Graph{
		shapes: [][]int{{3}},
		dtypes: []tensor.DataType{tensor.Float},
		kernel: &kernel.Kernel{},
	}
	g.kernel.Init()
	g.inputs = []int{g.kernel.RegisterWriter("input1")}

	input := []any{[]float32{1.0, 2.0, 3.0}, []float32{4.0, 5.0, 6.0}} // Extra input provided
	err := g.Execute(input)
	if err == nil {
		t.Errorf("Expected an error due to too many inputs, but got none")
	}
}

func TestExecute_ShapeMismatch(t *testing.T) {
	g := &Graph{
		shapes: [][]int{{2, 2}}, // Expecting a 2x2 shape
		dtypes: []tensor.DataType{tensor.Float},
		kernel: &kernel.Kernel{},
	}
	g.kernel.Init()
	g.inputs = []int{g.kernel.RegisterWriter("input1")}

	input := []any{[]float32{1.0, 2.0, 3.0}} // 1D input, cannot fit 2x2
	err := g.Execute(input)
	if err == nil {
		t.Errorf("Expected an error due to shape mismatch, but got none")
	}
}

func TestExecute_MismatchedShapeNegativeDim(t *testing.T) {
	g := &Graph{
		shapes: [][]int{{-1, 4}}, // Expecting a reshape to (-1,4)
		dtypes: []tensor.DataType{tensor.Float},
		kernel: &kernel.Kernel{},
	}
	g.kernel.Init()
	g.inputs = []int{g.kernel.RegisterWriter("input1")}

	input := []any{[]float32{1.0, 2.0, 3.0, 4.0, 5.0}} // Cannot reshape into (-1,4)
	err := g.Execute(input)
	if err == nil {
		t.Errorf("Expected an error due to shape mismatch in (-1,4) reshape, but got none")
	}
}

func TestExecute_ShapeMismatch_ExtraRows(t *testing.T) {
	g := &Graph{
		shapes: [][]int{{2, 3}}, // Expecting 2x3 shape
		dtypes: []tensor.DataType{tensor.Float},
		kernel: &kernel.Kernel{},
	}
	g.kernel.Init()
	g.inputs = []int{g.kernel.RegisterWriter("input1")}

	input := []any{[][]float32{{1.0, 2.0, 3.0}, {4.0, 5.0}}} // Second row has fewer columns
	err := g.Execute(input)
	if err == nil {
		t.Errorf("Expected an error due to inconsistent row sizes, but got none")
	}
}

func TestExecute_2DShapeMismatch_ColumnCount(t *testing.T) {
	g := &Graph{
		shapes: [][]int{{3, 2}}, // Expecting 3x2 shape
		dtypes: []tensor.DataType{tensor.Float},
		kernel: &kernel.Kernel{},
	}
	g.kernel.Init()
	g.inputs = []int{g.kernel.RegisterWriter("input1")}

	input := []any{[][]float32{{1.0, 2.0, 3.0}, {4.0, 5.0, 6.0}, {7.0, 8.0, 9.0}}} // Too many columns per row
	err := g.Execute(input)
	if err == nil {
		t.Errorf("Expected an error due to too many columns, but got none")
	}
}

func TestExecute_StaticValueShapeMismatch(t *testing.T) {
	g := &Graph{
		shapes: [][]int{{2}}, // Expecting a shape of 2, but static value can't fit
		dtypes: []tensor.DataType{tensor.Float},
		kernel: &kernel.Kernel{},
	}
	g.kernel.Init()
	g.inputs = []int{g.kernel.RegisterWriter("input1")}

	staticValue := float32(42.0)

	err := g.Execute([]any{staticValue})
	if err == nil {
		t.Errorf("Expected an error due to shape mismatch for static value, but got none")
	}
}

func BenchmarkExecute_LargeInput(b *testing.B) {
	g := &Graph{
		shapes: [][]int{{1000000}},
		dtypes: []tensor.DataType{tensor.Float},
		kernel: &kernel.Kernel{},
	}
	g.kernel.Init()
	g.inputs = []int{g.kernel.RegisterWriter("input1")}

	largeInput := make([]float32, 1000000)
	for i := range largeInput {
		largeInput[i] = float32(i)
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = g.Execute([]any{largeInput})
	}
}

func BenchmarkExecute_MultipleLargeInputs(b *testing.B) {
	g := &Graph{
		shapes: [][]int{{100000}, {100000}},
		dtypes: []tensor.DataType{tensor.Float, tensor.Int32},
		kernel: &kernel.Kernel{},
	}
	g.kernel.Init()
	g.inputs = []int{g.kernel.RegisterWriter("input1"), g.kernel.RegisterWriter("input2")}

	largeFloatInput := make([]float32, 100000)
	largeIntInput := make([]int32, 100000)
	for i := range largeFloatInput {
		largeFloatInput[i] = float32(i)
		largeIntInput[i] = int32(i)
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = g.Execute([]any{largeFloatInput, largeIntInput})
	}
}

func BenchmarkExecute_Large2DInput(b *testing.B) {
	g := &Graph{
		shapes: [][]int{{1000, 1000}},
		dtypes: []tensor.DataType{tensor.Float},
		kernel: &kernel.Kernel{},
	}
	g.kernel.Init()
	g.inputs = []int{g.kernel.RegisterWriter("input1")}

	large2DInput := make([][]float32, 1000)
	for i := range large2DInput {
		large2DInput[i] = make([]float32, 1000)
		for j := range large2DInput[i] {
			large2DInput[i][j] = float32(i * j)
		}
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = g.Execute([]any{large2DInput})
	}
}
