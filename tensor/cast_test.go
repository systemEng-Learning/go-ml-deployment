package tensor

import (
	"reflect"
	"testing"
)

// Test for casting from Float to Double
func TestCastFloatToDouble(t *testing.T) {
	tensor := &Tensor{
		DType:     Float,
		Shape:     []int{2, 3},
		FloatData: []float32{1, 2, 3, 4, 5, 6},
	}

	tensor.Cast(Double)

	expected := []float64{1, 2, 3, 4, 5, 6}
	if !reflect.DeepEqual(tensor.DoubleData, expected) {
		t.Errorf("Expected DoubleData to be %v, but got %v", expected, tensor.DoubleData)
	}
}

// Test for casting from Double to Int32
func TestCastDoubleToInt32(t *testing.T) {
	tensor := &Tensor{
		DType:      Double,
		Shape:      []int{2, 2},
		DoubleData: []float64{1.1, 2.9, 3.7, 4.4},
	}

	tensor.Cast(Int32)

	expected := []int32{1, 2, 3, 4}
	if !reflect.DeepEqual(tensor.Int32Data, expected) {
		t.Errorf("Expected Int32Data to be %v, but got %v", expected, tensor.Int32Data)
	}
}

// Test for casting from Int32 to Float
func TestCastInt32ToFloat(t *testing.T) {
	tensor := &Tensor{
		DType:     Int32,
		Shape:     []int{1, 3},
		Int32Data: []int32{1, 2, 3},
	}

	tensor.Cast(Float)

	expected := []float32{1.0, 2.0, 3.0}
	if !reflect.DeepEqual(tensor.FloatData, expected) {
		t.Errorf("Expected FloatData to be %v, but got %v", expected, tensor.FloatData)
	}
}

// Test for unsupported cast
func TestCastUnsupported(t *testing.T) {
	tensor := &Tensor{
		DType: IntMap,
		Shape: []int{2, 2},
	}

	tensor.Cast(Int32) // This should log an error since IntMap can't be cast

	// You can check the logs manually, or capture the log if needed for tests
}

func TestCastToSameType(t *testing.T) {
	tensor := &Tensor{
		DType:     Float,
		Shape:     []int{2, 3},
		FloatData: []float32{1.1, 2.2, 3.3, 4.4, 5.5, 6.6},
	}

	tensor.Cast(Float) // Casting to the same type should have no effect

	expected := []float32{1.1, 2.2, 3.3, 4.4, 5.5, 6.6}
	if !reflect.DeepEqual(tensor.FloatData, expected) {
		t.Errorf("Expected FloatData to be %v, but got %v", expected, tensor.FloatData)
	}
}

func TestCastIncorrectLength(t *testing.T) {
	tensor := &Tensor{
		DType:     Float,
		Shape:     []int{2, 2}, // Shape is 2x2, but data length is 5
		FloatData: []float32{1.1, 2.2, 3.3, 4.4, 5.5},
	}

	// This should fail because the data length doesn't match the shape
	tensor.Cast(Double) // This might cause an error in casting due to mismatched lengths

	expectedLen := 4
	if len(tensor.DoubleData) != 4 {
		t.Errorf("Expected DoubleData to be %d, but got %d", expectedLen, len(tensor.DoubleData))
	}
}

func TestCastFloatToInt32PrecisionLoss(t *testing.T) {
	tensor := &Tensor{
		DType:     Float,
		Shape:     []int{3},
		FloatData: []float32{1.9, 2.5, 3.1}, // Floats with potential precision loss when cast to Int32
	}

	tensor.Cast(Int32)

	expected := []int32{1, 2, 3} // Should truncate or round down
	if !reflect.DeepEqual(tensor.Int32Data, expected) {
		t.Errorf("Expected Int32Data to be %v, but got %v", expected, tensor.Int32Data)
	}
}
