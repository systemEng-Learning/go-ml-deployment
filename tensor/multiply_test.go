package tensor

import "testing"

func TestDot_Vector(t *testing.T) {
	a := &Tensor{FloatData: []float32{1, 2, 3}, Shape: []int{3}, DType: Float}
	b := &Tensor{FloatData: []float32{4, 5, 6}, Shape: []int{3}, DType: Float}

	result, err := a.Dot(b, nil)
	if err != nil {
		t.Fatalf("Unexpected error: %v", err)
	}

	expected := float32(32) // 1*4 + 2*5 + 3*6
	if result.FloatData[0] != expected {
		t.Errorf("Expected %v, got %v", expected, result.FloatData[0])
	}
}

func TestDot_MatrixVector(t *testing.T) {
	a := &Tensor{FloatData: []float32{1, 2, 3, 4}, Shape: []int{2, 2}, DType: Float}
	b := &Tensor{FloatData: []float32{5, 6}, Shape: []int{2}, DType: Float}

	result, err := a.Dot(b, nil)
	if err != nil {
		t.Fatalf("Unexpected error: %v", err)
	}

	expected := []float32{17, 39} // [1*5+2*6, 3*5+4*6]
	for i, v := range expected {
		if result.FloatData[i] != v {
			t.Errorf("Expected %v at index %d, got %v", v, i, result.FloatData[i])
		}
	}
}

func TestDot_MatrixMatrix(t *testing.T) {
	a := &Tensor{FloatData: []float32{1, 2, 3, 4, 5, 6}, Shape: []int{2, 3}, DType: Float}
	b := &Tensor{FloatData: []float32{7, 9, 11, 8, 10, 12}, Shape: []int{2, 3}, DType: Float}

	result, err := a.Dot(b, nil)
	if err != nil {
		t.Fatalf("Unexpected error: %v", err)
	}

	expected := []float32{58, 64, 139, 154}
	for i, v := range expected {
		if result.FloatData[i] != v {
			t.Errorf("Expected %v at index %d, got %v", v, i, result.FloatData[i])
		}
	}
}

func TestDot_IncompatibleShapes(t *testing.T) {
	a := &Tensor{FloatData: []float32{1, 2, 3}, Shape: []int{3}, DType: Float}
	b := &Tensor{FloatData: []float32{4, 5}, Shape: []int{2}, DType: Float}

	_, err := a.Dot(b, nil)
	if err == nil {
		t.Error("Expected error due to shape mismatch, got nil")
	}
}

func TestDot_Vector_DoubleOutput(t *testing.T) {
	a := &Tensor{FloatData: []float32{1, 2, 3}, Shape: []int{3}, DType: Float}
	b := &Tensor{DoubleData: []float64{4, 5, 6}, Shape: []int{3}, DType: Double}

	result, err := a.Dot(b, nil)
	if err != nil {
		t.Fatalf("Unexpected error: %v", err)
	}

	expected := float64(32) // 1*4 + 2*5 + 3*6
	if result.DoubleData[0] != expected {
		t.Errorf("Expected %v, got %v", expected, result.DoubleData[0])
	}
}

func TestDot_Vector_Int64Output(t *testing.T) {
	a := &Tensor{Int32Data: []int32{1, 2, 3}, Shape: []int{3}, DType: Int32}
	b := &Tensor{Int64Data: []int64{4, 5, 6}, Shape: []int{3}, DType: Int64}

	result, err := a.Dot(b, nil)
	if err != nil {
		t.Fatalf("Unexpected error: %v", err)
	}

	expected := int64(32) // 1*4 + 2*5 + 3*6
	if result.Int64Data[0] != expected {
		t.Errorf("Expected %v, got %v", expected, result.Int64Data[0])
	}
}

func TestDot_Vector_Double(t *testing.T) {
	a := &Tensor{DoubleData: []float64{1, 2, 3}, Shape: []int{3}, DType: Double}
	b := &Tensor{DoubleData: []float64{4, 5, 6}, Shape: []int{3}, DType: Double}

	result, err := a.Dot(b, nil)
	if err != nil {
		t.Fatalf("Unexpected error: %v", err)
	}

	expected := float64(32) // 1*4 + 2*5 + 3*6
	if result.DoubleData[0] != expected {
		t.Errorf("Expected %v, got %v", expected, result.DoubleData[0])
	}
}

func TestDot_MatrixVector_Double(t *testing.T) {
	a := &Tensor{DoubleData: []float64{1, 2, 3, 4}, Shape: []int{2, 2}, DType: Double}
	b := &Tensor{DoubleData: []float64{5, 6}, Shape: []int{2}, DType: Double}

	result, err := a.Dot(b, nil)
	if err != nil {
		t.Fatalf("Unexpected error: %v", err)
	}

	expected := []float64{17, 39} // [1*5+2*6, 3*5+4*6]
	for i, v := range expected {
		if result.DoubleData[i] != v {
			t.Errorf("Expected %v at index %d, got %v", v, i, result.DoubleData[i])
		}
	}
}

func TestDot_MatrixMatrix_Double(t *testing.T) {
	a := &Tensor{DoubleData: []float64{1, 2, 3, 4, 5, 6}, Shape: []int{2, 3}, DType: Double}
	b := &Tensor{DoubleData: []float64{7, 9, 11, 8, 10, 12}, Shape: []int{2, 3}, DType: Double}

	result, err := a.Dot(b, nil)
	if err != nil {
		t.Fatalf("Unexpected error: %v", err)
	}

	expected := []float64{58, 64, 139, 154}
	for i, v := range expected {
		if result.DoubleData[i] != v {
			t.Errorf("Expected %v at index %d, got %v", v, i, result.DoubleData[i])
		}
	}
}

func TestDot_Vector_Int32(t *testing.T) {
	a := &Tensor{Int32Data: []int32{1, 2, 3}, Shape: []int{3}, DType: Int32}
	b := &Tensor{Int32Data: []int32{4, 5, 6}, Shape: []int{3}, DType: Int32}

	result, err := a.Dot(b, nil)
	if err != nil {
		t.Fatalf("Unexpected error: %v", err)
	}

	expected := int32(32) // 1*4 + 2*5 + 3*6
	if result.Int32Data[0] != expected {
		t.Errorf("Expected %v, got %v", expected, result.Int32Data[0])
	}
}

func TestDot_MatrixVector_Int32(t *testing.T) {
	a := &Tensor{Int32Data: []int32{1, 2, 3, 4}, Shape: []int{2, 2}, DType: Int32}
	b := &Tensor{Int32Data: []int32{5, 6}, Shape: []int{2}, DType: Int32}

	result, err := a.Dot(b, nil)
	if err != nil {
		t.Fatalf("Unexpected error: %v", err)
	}

	expected := []int32{17, 39} // [1*5+2*6, 3*5+4*6]
	for i, v := range expected {
		if result.Int32Data[i] != v {
			t.Errorf("Expected %v at index %d, got %v", v, i, result.Int32Data[i])
		}
	}
}

func TestDot_MatrixMatrix_Int32(t *testing.T) {
	a := &Tensor{Int32Data: []int32{1, 2, 3, 4, 5, 6}, Shape: []int{2, 3}, DType: Int32}
	b := &Tensor{Int32Data: []int32{7, 9, 11, 8, 10, 12}, Shape: []int{2, 3}, DType: Int32}

	result, err := a.Dot(b, nil)
	if err != nil {
		t.Fatalf("Unexpected error: %v", err)
	}

	expected := []int32{58, 64, 139, 154}
	for i, v := range expected {
		if result.Int32Data[i] != v {
			t.Errorf("Expected %v at index %d, got %v", v, i, result.Int32Data[i])
		}
	}
}

func TestDot_Vector_Int64(t *testing.T) {
	a := &Tensor{Int64Data: []int64{1, 2, 3}, Shape: []int{3}, DType: Int64}
	b := &Tensor{Int64Data: []int64{4, 5, 6}, Shape: []int{3}, DType: Int64}

	result, err := a.Dot(b, nil)
	if err != nil {
		t.Fatalf("Unexpected error: %v", err)
	}

	expected := int64(32) // 1*4 + 2*5 + 3*6
	if result.Int64Data[0] != expected {
		t.Errorf("Expected %v, got %v", expected, result.Int64Data[0])
	}
}

func TestDot_MatrixVector_Int64(t *testing.T) {
	a := &Tensor{Int64Data: []int64{1, 2, 3, 4}, Shape: []int{2, 2}, DType: Int64}
	b := &Tensor{Int64Data: []int64{5, 6}, Shape: []int{2}, DType: Int64}

	result, err := a.Dot(b, nil)
	if err != nil {
		t.Fatalf("Unexpected error: %v", err)
	}

	expected := []int64{17, 39} // [1*5+2*6, 3*5+4*6]
	for i, v := range expected {
		if result.Int64Data[i] != v {
			t.Errorf("Expected %v at index %d, got %v", v, i, result.Int64Data[i])
		}
	}
}

func TestDot_MatrixMatrix_Int64(t *testing.T) {
	a := &Tensor{Int64Data: []int64{1, 2, 3, 4, 5, 6}, Shape: []int{2, 3}, DType: Int64}
	b := &Tensor{Int64Data: []int64{7, 9, 11, 8, 10, 12}, Shape: []int{2, 3}, DType: Int64}

	result, err := a.Dot(b, nil)
	if err != nil {
		t.Fatalf("Unexpected error: %v", err)
	}

	expected := []int64{58, 64, 139, 154}
	for i, v := range expected {
		if result.Int64Data[i] != v {
			t.Errorf("Expected %v at index %d, got %v", v, i, result.Int64Data[i])
		}
	}
}

func TestDot_Vector_Int32FloatInputs(t *testing.T) {
	a := &Tensor{Int32Data: []int32{1, 2, 3}, Shape: []int{3}, DType: Int32}
	b := &Tensor{FloatData: []float32{4, 5, 6}, Shape: []int{3}, DType: Float}

	result, err := a.Dot(b, nil)
	if err != nil {
		t.Fatalf("Unexpected error: %v", err)
	}

	expected := float64(32) // 1*4 + 2*5 + 3*6
	if result.DoubleData[0] != expected {
		t.Errorf("Expected %v, got %v", expected, result.DoubleData[0])
	}
}
