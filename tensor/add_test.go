package tensor

import (
	"reflect"
	"testing"
)

func TestAdd_SameShape(t *testing.T) {
	tests := []struct {
		name     string
		a        *Tensor
		b        *Tensor
		outDType DataType
		expected any
	}{
		{
			name:     "Float + Float (1D)",
			a:        Create1DFloatTensor([]float32{1, 2, 3}),
			b:        Create1DFloatTensor([]float32{4, 5, 6}),
			outDType: Float,
			expected: []float32{5, 7, 9},
		},
		{
			name:     "Int32 + Int32 (1D)",
			a:        mustTensor(CreateEmptyTensor([]int{3}, Int32), []int32{1, 2, 3}),
			b:        mustTensor(CreateEmptyTensor([]int{3}, Int32), []int32{4, 5, 6}),
			outDType: Int32,
			expected: []int32{5, 7, 9},
		},
		{
			name:     "Float + Float (2D)",
			a:        mustTensor(CreateEmptyTensor([]int{2, 2}, Float), []float32{1, 2, 3, 4}),
			b:        mustTensor(CreateEmptyTensor([]int{2, 2}, Float), []float32{5, 6, 7, 8}),
			outDType: Float,
			expected: []float32{6, 8, 10, 12},
		},
		{
			name:     "Int64 + Int64 (1D)",
			a:        mustTensor(CreateEmptyTensor([]int{3}, Int64), []int64{100, 200, 300}),
			b:        mustTensor(CreateEmptyTensor([]int{3}, Int64), []int64{10, 20, 30}),
			outDType: Int64,
			expected: []int64{110, 220, 330},
		},
		{
			name:     "Int32 + Float (1D)",
			a:        mustTensor(CreateEmptyTensor([]int{3}, Int32), []int32{1, 2, 3}),
			b:        Create1DFloatTensor([]float32{4, 5, 6}),
			outDType: Double,
			expected: []float64{5, 7, 9},
		},
		{
			name:     "Int32 + Int64 (1D)",
			a:        mustTensor(CreateEmptyTensor([]int{3}, Int32), []int32{1, 2, 3}),
			b:        mustTensor(CreateEmptyTensor([]int{3}, Int64), []int64{4, 5, 6}),
			outDType: Int64,
			expected: []int64{5, 7, 9},
		},
		{
			name:     "Float + Double (2D)",
			a:        mustTensor(CreateEmptyTensor([]int{2, 2}, Float), []float32{1, 2, 3, 4}),
			b:        mustTensor(CreateEmptyTensor([]int{2, 2}, Double), []float64{5, 6, 7, 8}),
			outDType: Double,
			expected: []float64{6, 8, 10, 12},
		},
		{
			name:     "Int64 + Double (1D)",
			a:        mustTensor(CreateEmptyTensor([]int{3}, Int64), []int64{100, 200, 300}),
			b:        mustTensor(CreateEmptyTensor([]int{3}, Double), []float64{10, 20, 30}),
			outDType: Double,
			expected: []float64{110, 220, 330},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			out, err := tt.a.Add(tt.b, nil)
			if err != nil {
				t.Errorf("Add() error: %v", err)
				return
			}
			if out.DType != tt.outDType {
				t.Errorf("expected DType %v, got %v", tt.outDType, out.DType)
			}
			if !reflect.DeepEqual(extractData(out), tt.expected) {
				t.Errorf("expected data %v, got %v", tt.expected, extractData(out))
			}
		})
	}
}

func TestAdd_ElemBroadcast(t *testing.T) {
	a := Create1DFloatTensor([]float32{1, 2, 3})
	b := Create1DFloatTensor([]float32{10}) // scalar
	expected := []float32{11, 12, 13}

	out, err := a.Add(b, nil)
	if err != nil {
		t.Fatalf("Add() error: %v", err)
	}
	if !reflect.DeepEqual(out.FloatData, expected) {
		t.Errorf("expected %v, got %v", expected, out.FloatData)
	}
}

func TestAdd_RowBroadcast(t *testing.T) {
	a := mustTensor(CreateEmptyTensor([]int{2, 3}, Float), []float32{1, 2, 3, 4, 5, 6})
	b := mustTensor(CreateEmptyTensor([]int{1, 3}, Float), []float32{10, 20, 30})
	expected := []float32{11, 22, 33, 14, 25, 36}

	out, err := a.Add(b, nil)
	if err != nil {
		t.Fatalf("Add() error: %v", err)
	}
	if !reflect.DeepEqual(out.FloatData, expected) {
		t.Errorf("expected %v, got %v", expected, out.FloatData)
	}
}

func TestAdd_ColBroadcast(t *testing.T) {
	a := mustTensor(CreateEmptyTensor([]int{2, 3}, Float), []float32{1, 2, 3, 4, 5, 6})
	b := mustTensor(CreateEmptyTensor([]int{2, 1}, Float), []float32{10, 100})
	expected := []float32{11, 12, 13, 104, 105, 106}

	out, err := a.Add(b, nil)
	if err != nil {
		t.Fatalf("Add() error: %v", err)
	}
	if !reflect.DeepEqual(out.FloatData, expected) {
		t.Errorf("expected %v, got %v", expected, out.FloatData)
	}
}
