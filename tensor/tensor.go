package tensors

import (
	"fmt"
	"maps"
	"slices"
)

type DataType int

const (
	Undefined DataType = iota
	Float
	Int32
	Int64
	Double
	StringMap
	IntMap
)

type Tensor struct {
	Shape      []int
	DType      DataType
	FloatData  []float32
	Int32Data  []int32
	Int64Data  []int64
	DoubleData []float64
	IntMap     map[int]float32
	StringMap  map[string]float32
}

func (t *Tensor) Copy() (Tensor, error) {
	newTensor := Tensor{}
	switch t.DType {
	case Float:
		newTensor.FloatData = slices.Clone(t.FloatData)
	case Int32:
		newTensor.Int32Data = slices.Clone(t.Int32Data)
	case Int64:
		newTensor.Int64Data = slices.Clone(t.Int64Data)
	case Double:
		newTensor.DoubleData = slices.Clone(t.DoubleData)
	case IntMap:
		newTensor.IntMap = maps.Clone(t.IntMap)
	case StringMap:
		newTensor.StringMap = maps.Clone(t.StringMap)
	default:
		return newTensor, fmt.Errorf("tensor copy: unsupported data type %d", t.DType)
	}
	newTensor.DType = t.DType
	newTensor.Shape = slices.Clone(t.Shape)
	return newTensor, nil
}

func CreateTensor(shape []int, dataType DataType) (*Tensor, error) {
	if dataType == Undefined {
		return nil, fmt.Errorf("create tensor: unsupported data type")
	}
	t := &Tensor{
		Shape: shape,
		DType: dataType,
	}

	switch dataType {
	case Float:
		t.FloatData = make([]float32, shape[0]*shape[1])
	case Int32:
		t.Int32Data = make([]int32, shape[0]*shape[1])
	case Int64:
		t.Int64Data = make([]int64, shape[0]*shape[1])
	case Double:
		t.DoubleData = make([]float64, shape[0]*shape[1])
	case IntMap:
		t.IntMap = make(map[int]float32)
	case StringMap:
		t.StringMap = make(map[string]float32)
	}
	return t, nil
}

func OnnxTypeToDtype(elemType string) DataType {
	switch elemType {
	case "FLOAT":
		return Float
	case "INT32":
		return Int32
	case "INT64":
		return Int64
	case "DOUBLE":
		return Double
	default:
		return Undefined
	}
}
