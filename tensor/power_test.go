package tensor

import (
	"reflect"
	"testing"
)

func TestSquare(t *testing.T) {
	a := Create1DFloatTensor([]float32{1, 2, 3})
	expected := []float32{1, 4, 9}

	a.Square()
	if !reflect.DeepEqual(a.FloatData, expected) {
		t.Errorf("expected %v, got %v", expected, a.FloatData)
	}
}

func TestCube(t *testing.T) {
	a := Create1DFloatTensor([]float32{1, 2, 3})
	expected := []float32{1, 8, 27}

	a.Cube()
	if !reflect.DeepEqual(a.FloatData, expected) {
		t.Errorf("expected %v, got %v", expected, a.FloatData)
	}
}

func TestRaiseTo5(t *testing.T) {
	a := Create1DFloatTensor([]float32{1, 2, 3})
	expected := []float32{1, 32, 243}

	a.Power(5)
	if !reflect.DeepEqual(a.FloatData, expected) {
		t.Errorf("expected %v, got %v", expected, a.FloatData)
	}
}

func TestSquare2D(t *testing.T) {
	a := mustTensor(CreateEmptyTensor([]int{2, 3}, Float), []float32{1, 2, 3, 4, 5, 6})
	expected := []float32{1, 4, 9, 16, 25, 36}

	a.Square()
	if !reflect.DeepEqual(a.FloatData, expected) {
		t.Errorf("expected %v, got %v", expected, a.FloatData)
	}
}

func TestCube2D(t *testing.T) {
	a := mustTensor(CreateEmptyTensor([]int{2, 3}, Float), []float32{1, 2, 3, 4, 5, 6})
	expected := []float32{1, 8, 27, 64, 125, 216}

	a.Cube()
	if !reflect.DeepEqual(a.FloatData, expected) {
		t.Errorf("expected %v, got %v", expected, a.FloatData)
	}
}

func TestRaiseTo42D(t *testing.T) {
	a := mustTensor(CreateEmptyTensor([]int{2, 3}, Float), []float32{1, 2, 3, 4, 5, 6})
	expected := []float32{1, 16, 81, 256, 625, 1296}

	a.Power(4)
	if !reflect.DeepEqual(a.FloatData, expected) {
		t.Errorf("expected %v, got %v", expected, a.FloatData)
	}
}
