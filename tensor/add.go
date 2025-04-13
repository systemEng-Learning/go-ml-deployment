package tensor

import (
	"errors"
	"fmt"
)

type Numeric interface {
	int32 | int64 | float32 | float64
}

type addKey struct {
	from1, from2, out DataType
}

func (t *Tensor) Add(other *Tensor, out *Tensor) (*Tensor, error) {
	// Cannot add maps
	if t.DType == IntMap || t.DType == StringMap || other.DType == IntMap || other.DType == StringMap {
		return nil, errors.New("cannot execute dot operations on map tensors")
	}
	if len(t.Shape) == len(other.Shape) {
		if (len(t.Shape) == 1 && t.Shape[0] == other.Shape[0]) ||
			(len(t.Shape) == 2 && t.Shape[0] == other.Shape[0] && t.Shape[1] == other.Shape[1]) {
			return t.sameShapeAdd(other, out)
		}
		if t.Shape[0] == 1 && (len(t.Shape) == 1 || (len(t.Shape) == 2 && t.Shape[1] == 1)) {
			return other.addElem(t, out)
		}
		if other.Shape[0] == 1 && (len(other.Shape) == 1 || (len(other.Shape) == 2 && other.Shape[1] == 1)) {
			return t.addElem(other, out)
		}
		if len(t.Shape) == 1 && t.Shape[0] != other.Shape[0] {
			return nil, fmt.Errorf("both vectors should have equal length %d != %d", t.Shape[0], other.Shape[0])
		}
		if t.Shape[0] == 1 {
			return other.addRow(t, out)
		}
		if other.Shape[0] == 1 {
			return t.addRow(other, out)
		}
		if t.Shape[1] == 1 {
			return other.addCol(t, out)
		}
		if other.Shape[1] == 1 {
			return t.addCol(other, out)
		}
		return nil, fmt.Errorf("both matrixes should have the same shape %v != %v", t.Shape, other.Shape)
	} else if len(t.Shape) == 2 {
		if other.Shape[0] == 1 {
			return t.addElem(other, out)
		}
		if other.Shape[0] == t.Shape[1] {
			return t.addRow(other, out)
		}
		if other.Shape[0] == t.Shape[0] {
			return t.addCol(other, out)
		}
		return nil, fmt.Errorf("operants shape do not align %v != %v", t.Shape, other.Shape)
	} else if len(other.Shape) == 2 {
		if t.Shape[0] == 1 {
			return other.addElem(t, out)
		}
		if t.Shape[0] == other.Shape[1] {
			return other.addRow(t, out)
		}
		if t.Shape[0] == other.Shape[0] {
			return other.addCol(t, out)
		}
		return nil, fmt.Errorf("operants shape do not align %v != %v", t.Shape, other.Shape)
	}

	return nil, fmt.Errorf("cannot add the 2 tensors shapes do not align %v != %v", t.Shape, other.Shape)
}

type opAddFunc func(a, b, out any, length int)

var opAddDispatch = map[addKey]opAddFunc{
	{Float, Float, Float}: func(a, b, out any, l int) {
		OpAdd(a.([]float32), b.([]float32), out.([]float32), l)
	},
	{Float, Float, Double}: func(a, b, out any, l int) {
		OpAdd(a.([]float32), b.([]float32), out.([]float64), l)
	},
	{Float, Float, Int32}: func(a, b, out any, l int) {
		OpAdd(a.([]float32), b.([]float32), out.([]int32), l)
	},
	{Float, Float, Int64}: func(a, b, out any, l int) {
		OpAdd(a.([]float32), b.([]float32), out.([]int64), l)
	},
	{Float, Double, Float}: func(a, b, out any, l int) {
		OpAdd(a.([]float32), b.([]float64), out.([]float32), l)
	},
	{Float, Double, Double}: func(a, b, out any, l int) {
		OpAdd(a.([]float32), b.([]float64), out.([]float64), l)
	},
	{Float, Double, Int32}: func(a, b, out any, l int) {
		OpAdd(a.([]float32), b.([]float64), out.([]int32), l)
	},
	{Float, Double, Int64}: func(a, b, out any, l int) {
		OpAdd(a.([]float32), b.([]float64), out.([]int64), l)
	},
	{Float, Int32, Float}: func(a, b, out any, l int) {
		OpAdd(a.([]float32), b.([]int32), out.([]float32), l)
	},
	{Float, Int32, Double}: func(a, b, out any, l int) {
		OpAdd(a.([]float32), b.([]int32), out.([]float64), l)
	},
	{Float, Int32, Int32}: func(a, b, out any, l int) {
		OpAdd(a.([]float32), b.([]int32), out.([]int32), l)
	},
	{Float, Int32, Int64}: func(a, b, out any, l int) {
		OpAdd(a.([]float32), b.([]int32), out.([]int64), l)
	},
	{Float, Int64, Float}: func(a, b, out any, l int) {
		OpAdd(a.([]float32), b.([]int64), out.([]float32), l)
	},
	{Float, Int64, Double}: func(a, b, out any, l int) {
		OpAdd(a.([]float32), b.([]int64), out.([]float64), l)
	},
	{Float, Int64, Int32}: func(a, b, out any, l int) {
		OpAdd(a.([]float32), b.([]int64), out.([]int32), l)
	},
	{Float, Int64, Int64}: func(a, b, out any, l int) {
		OpAdd(a.([]float32), b.([]int64), out.([]int64), l)
	},
	{Double, Float, Float}: func(a, b, out any, l int) {
		OpAdd(a.([]float64), b.([]float32), out.([]float32), l)
	},
	{Double, Float, Double}: func(a, b, out any, l int) {
		OpAdd(a.([]float64), b.([]float32), out.([]float64), l)
	},
	{Double, Float, Int32}: func(a, b, out any, l int) {
		OpAdd(a.([]float64), b.([]float32), out.([]int32), l)
	},
	{Double, Float, Int64}: func(a, b, out any, l int) {
		OpAdd(a.([]float64), b.([]float32), out.([]int64), l)
	},
	{Double, Double, Float}: func(a, b, out any, l int) {
		OpAdd(a.([]float64), b.([]float64), out.([]float32), l)
	},
	{Double, Double, Double}: func(a, b, out any, l int) {
		OpAdd(a.([]float64), b.([]float64), out.([]float64), l)
	},
	{Double, Double, Int32}: func(a, b, out any, l int) {
		OpAdd(a.([]float64), b.([]float64), out.([]int32), l)
	},
	{Double, Double, Int64}: func(a, b, out any, l int) {
		OpAdd(a.([]float64), b.([]float64), out.([]int64), l)
	},
	{Double, Int32, Float}: func(a, b, out any, l int) {
		OpAdd(a.([]float64), b.([]int32), out.([]float32), l)
	},
	{Double, Int32, Double}: func(a, b, out any, l int) {
		OpAdd(a.([]float64), b.([]int32), out.([]float64), l)
	},
	{Double, Int32, Int32}: func(a, b, out any, l int) {
		OpAdd(a.([]float64), b.([]int32), out.([]int32), l)
	},
	{Double, Int32, Int64}: func(a, b, out any, l int) {
		OpAdd(a.([]float64), b.([]int32), out.([]int64), l)
	},
	{Double, Int64, Float}: func(a, b, out any, l int) {
		OpAdd(a.([]float64), b.([]int64), out.([]float32), l)
	},
	{Double, Int64, Double}: func(a, b, out any, l int) {
		OpAdd(a.([]float64), b.([]int64), out.([]float64), l)
	},
	{Double, Int64, Int32}: func(a, b, out any, l int) {
		OpAdd(a.([]float64), b.([]int64), out.([]int32), l)
	},
	{Double, Int64, Int64}: func(a, b, out any, l int) {
		OpAdd(a.([]float64), b.([]int64), out.([]int64), l)
	},
	{Int32, Float, Float}: func(a, b, out any, l int) {
		OpAdd(a.([]int32), b.([]float32), out.([]float32), l)
	},
	{Int32, Float, Double}: func(a, b, out any, l int) {
		OpAdd(a.([]int32), b.([]float32), out.([]float64), l)
	},
	{Int32, Float, Int32}: func(a, b, out any, l int) {
		OpAdd(a.([]int32), b.([]float32), out.([]int32), l)
	},
	{Int32, Float, Int64}: func(a, b, out any, l int) {
		OpAdd(a.([]int32), b.([]float32), out.([]int64), l)
	},
	{Int32, Double, Float}: func(a, b, out any, l int) {
		OpAdd(a.([]int32), b.([]float64), out.([]float32), l)
	},
	{Int32, Double, Double}: func(a, b, out any, l int) {
		OpAdd(a.([]int32), b.([]float64), out.([]float64), l)
	},
	{Int32, Double, Int32}: func(a, b, out any, l int) {
		OpAdd(a.([]int32), b.([]float64), out.([]int32), l)
	},
	{Int32, Double, Int64}: func(a, b, out any, l int) {
		OpAdd(a.([]int32), b.([]float64), out.([]int64), l)
	},
	{Int32, Int32, Float}: func(a, b, out any, l int) {
		OpAdd(a.([]int32), b.([]int32), out.([]float32), l)
	},
	{Int32, Int32, Double}: func(a, b, out any, l int) {
		OpAdd(a.([]int32), b.([]int32), out.([]float64), l)
	},
	{Int32, Int32, Int32}: func(a, b, out any, l int) {
		OpAdd(a.([]int32), b.([]int32), out.([]int32), l)
	},
	{Int32, Int32, Int64}: func(a, b, out any, l int) {
		OpAdd(a.([]int32), b.([]int32), out.([]int64), l)
	},
	{Int32, Int64, Float}: func(a, b, out any, l int) {
		OpAdd(a.([]int32), b.([]int64), out.([]float32), l)
	},
	{Int32, Int64, Double}: func(a, b, out any, l int) {
		OpAdd(a.([]int32), b.([]int64), out.([]float64), l)
	},
	{Int32, Int64, Int32}: func(a, b, out any, l int) {
		OpAdd(a.([]int32), b.([]int64), out.([]int32), l)
	},
	{Int32, Int64, Int64}: func(a, b, out any, l int) {
		OpAdd(a.([]int32), b.([]int64), out.([]int64), l)
	},
	{Int64, Float, Float}: func(a, b, out any, l int) {
		OpAdd(a.([]int64), b.([]float32), out.([]float32), l)
	},
	{Int64, Float, Double}: func(a, b, out any, l int) {
		OpAdd(a.([]int64), b.([]float32), out.([]float64), l)
	},
	{Int64, Float, Int32}: func(a, b, out any, l int) {
		OpAdd(a.([]int64), b.([]float32), out.([]int32), l)
	},
	{Int64, Float, Int64}: func(a, b, out any, l int) {
		OpAdd(a.([]int64), b.([]float32), out.([]int64), l)
	},
	{Int64, Double, Float}: func(a, b, out any, l int) {
		OpAdd(a.([]int64), b.([]float64), out.([]float32), l)
	},
	{Int64, Double, Double}: func(a, b, out any, l int) {
		OpAdd(a.([]int64), b.([]float64), out.([]float64), l)
	},
	{Int64, Double, Int32}: func(a, b, out any, l int) {
		OpAdd(a.([]int64), b.([]float64), out.([]int32), l)
	},
	{Int64, Double, Int64}: func(a, b, out any, l int) {
		OpAdd(a.([]int64), b.([]float64), out.([]int64), l)
	},
	{Int64, Int32, Float}: func(a, b, out any, l int) {
		OpAdd(a.([]int64), b.([]int32), out.([]float32), l)
	},
	{Int64, Int32, Double}: func(a, b, out any, l int) {
		OpAdd(a.([]int64), b.([]int32), out.([]float64), l)
	},
	{Int64, Int32, Int32}: func(a, b, out any, l int) {
		OpAdd(a.([]int64), b.([]int32), out.([]int32), l)
	},
	{Int64, Int32, Int64}: func(a, b, out any, l int) {
		OpAdd(a.([]int64), b.([]int32), out.([]int64), l)
	},
	{Int64, Int64, Float}: func(a, b, out any, l int) {
		OpAdd(a.([]int64), b.([]int64), out.([]float32), l)
	},
	{Int64, Int64, Double}: func(a, b, out any, l int) {
		OpAdd(a.([]int64), b.([]int64), out.([]float64), l)
	},
	{Int64, Int64, Int32}: func(a, b, out any, l int) {
		OpAdd(a.([]int64), b.([]int64), out.([]int32), l)
	},
	{Int64, Int64, Int64}: func(a, b, out any, l int) {
		OpAdd(a.([]int64), b.([]int64), out.([]int64), l)
	},
}

func OpAdd[T Numeric, U Numeric, V Numeric](first []T, second []U, result []V, length int) {
	for i := range length {
		result[i] = V(first[i]) + V(second[i])
	}
}

func (t *Tensor) sameShapeAdd(other *Tensor, out *Tensor) (*Tensor, error) {
	if out == nil {
		out = createOutputTensor(t.DType, other.DType, t.Shape)
	}

	length := t.Shape[0]
	if len(t.Shape) == 2 {
		length *= t.Shape[1]
	}

	key := addKey{t.DType, other.DType, out.DType}
	fn, ok := opAddDispatch[key]
	if !ok {
		return nil, fmt.Errorf("unsupported add combination: %v + %v -> %v", t.DType, other.DType, out.DType)
	}

	fn(t.rawData(), other.rawData(), out.rawData(), length)
	return out, nil
}

var elemAddDispatch = map[addKey]opAddFunc{
	{Float, Float, Float}: func(a, b, out any, l int) {
		ElemAdd(a.([]float32), b.([]float32)[0], out.([]float32), l)
	},
	{Float, Float, Double}: func(a, b, out any, l int) {
		ElemAdd(a.([]float32), b.([]float32)[0], out.([]float64), l)
	},
	{Float, Float, Int32}: func(a, b, out any, l int) {
		ElemAdd(a.([]float32), b.([]float32)[0], out.([]int32), l)
	},
	{Float, Float, Int64}: func(a, b, out any, l int) {
		ElemAdd(a.([]float32), b.([]float32)[0], out.([]int64), l)
	},
	{Float, Double, Float}: func(a, b, out any, l int) {
		ElemAdd(a.([]float32), b.([]float64)[0], out.([]float32), l)
	},
	{Float, Double, Double}: func(a, b, out any, l int) {
		ElemAdd(a.([]float32), b.([]float64)[0], out.([]float64), l)
	},
	{Float, Double, Int32}: func(a, b, out any, l int) {
		ElemAdd(a.([]float32), b.([]float64)[0], out.([]int32), l)
	},
	{Float, Double, Int64}: func(a, b, out any, l int) {
		ElemAdd(a.([]float32), b.([]float64)[0], out.([]int64), l)
	},
	{Float, Int32, Float}: func(a, b, out any, l int) {
		ElemAdd(a.([]float32), b.([]int32)[0], out.([]float32), l)
	},
	{Float, Int32, Double}: func(a, b, out any, l int) {
		ElemAdd(a.([]float32), b.([]int32)[0], out.([]float64), l)
	},
	{Float, Int32, Int32}: func(a, b, out any, l int) {
		ElemAdd(a.([]float32), b.([]int32)[0], out.([]int32), l)
	},
	{Float, Int32, Int64}: func(a, b, out any, l int) {
		ElemAdd(a.([]float32), b.([]int32)[0], out.([]int64), l)
	},
	{Float, Int64, Float}: func(a, b, out any, l int) {
		ElemAdd(a.([]float32), b.([]int64)[0], out.([]float32), l)
	},
	{Float, Int64, Double}: func(a, b, out any, l int) {
		ElemAdd(a.([]float32), b.([]int64)[0], out.([]float64), l)
	},
	{Float, Int64, Int32}: func(a, b, out any, l int) {
		ElemAdd(a.([]float32), b.([]int64)[0], out.([]int32), l)
	},
	{Float, Int64, Int64}: func(a, b, out any, l int) {
		ElemAdd(a.([]float32), b.([]int64)[0], out.([]int64), l)
	},
	{Double, Float, Float}: func(a, b, out any, l int) {
		ElemAdd(a.([]float64), b.([]float32)[0], out.([]float32), l)
	},
	{Double, Float, Double}: func(a, b, out any, l int) {
		ElemAdd(a.([]float64), b.([]float32)[0], out.([]float64), l)
	},
	{Double, Float, Int32}: func(a, b, out any, l int) {
		ElemAdd(a.([]float64), b.([]float32)[0], out.([]int32), l)
	},
	{Double, Float, Int64}: func(a, b, out any, l int) {
		ElemAdd(a.([]float64), b.([]float32)[0], out.([]int64), l)
	},
	{Double, Double, Float}: func(a, b, out any, l int) {
		ElemAdd(a.([]float64), b.([]float64)[0], out.([]float32), l)
	},
	{Double, Double, Double}: func(a, b, out any, l int) {
		ElemAdd(a.([]float64), b.([]float64)[0], out.([]float64), l)
	},
	{Double, Double, Int32}: func(a, b, out any, l int) {
		ElemAdd(a.([]float64), b.([]float64)[0], out.([]int32), l)
	},
	{Double, Double, Int64}: func(a, b, out any, l int) {
		ElemAdd(a.([]float64), b.([]float64)[0], out.([]int64), l)
	},
	{Double, Int32, Float}: func(a, b, out any, l int) {
		ElemAdd(a.([]float64), b.([]int32)[0], out.([]float32), l)
	},
	{Double, Int32, Double}: func(a, b, out any, l int) {
		ElemAdd(a.([]float64), b.([]int32)[0], out.([]float64), l)
	},
	{Double, Int32, Int32}: func(a, b, out any, l int) {
		ElemAdd(a.([]float64), b.([]int32)[0], out.([]int32), l)
	},
	{Double, Int32, Int64}: func(a, b, out any, l int) {
		ElemAdd(a.([]float64), b.([]int32)[0], out.([]int64), l)
	},
	{Double, Int64, Float}: func(a, b, out any, l int) {
		ElemAdd(a.([]float64), b.([]int64)[0], out.([]float32), l)
	},
	{Double, Int64, Double}: func(a, b, out any, l int) {
		ElemAdd(a.([]float64), b.([]int64)[0], out.([]float64), l)
	},
	{Double, Int64, Int32}: func(a, b, out any, l int) {
		ElemAdd(a.([]float64), b.([]int64)[0], out.([]int32), l)
	},
	{Double, Int64, Int64}: func(a, b, out any, l int) {
		ElemAdd(a.([]float64), b.([]int64)[0], out.([]int64), l)
	},
	{Int32, Float, Float}: func(a, b, out any, l int) {
		ElemAdd(a.([]int32), b.([]float32)[0], out.([]float32), l)
	},
	{Int32, Float, Double}: func(a, b, out any, l int) {
		ElemAdd(a.([]int32), b.([]float32)[0], out.([]float64), l)
	},
	{Int32, Float, Int32}: func(a, b, out any, l int) {
		ElemAdd(a.([]int32), b.([]float32)[0], out.([]int32), l)
	},
	{Int32, Float, Int64}: func(a, b, out any, l int) {
		ElemAdd(a.([]int32), b.([]float32)[0], out.([]int64), l)
	},
	{Int32, Double, Float}: func(a, b, out any, l int) {
		ElemAdd(a.([]int32), b.([]float64)[0], out.([]float32), l)
	},
	{Int32, Double, Double}: func(a, b, out any, l int) {
		ElemAdd(a.([]int32), b.([]float64)[0], out.([]float64), l)
	},
	{Int32, Double, Int32}: func(a, b, out any, l int) {
		ElemAdd(a.([]int32), b.([]float64)[0], out.([]int32), l)
	},
	{Int32, Double, Int64}: func(a, b, out any, l int) {
		ElemAdd(a.([]int32), b.([]float64)[0], out.([]int64), l)
	},
	{Int32, Int32, Float}: func(a, b, out any, l int) {
		ElemAdd(a.([]int32), b.([]int32)[0], out.([]float32), l)
	},
	{Int32, Int32, Double}: func(a, b, out any, l int) {
		ElemAdd(a.([]int32), b.([]int32)[0], out.([]float64), l)
	},
	{Int32, Int32, Int32}: func(a, b, out any, l int) {
		ElemAdd(a.([]int32), b.([]int32)[0], out.([]int32), l)
	},
	{Int32, Int32, Int64}: func(a, b, out any, l int) {
		ElemAdd(a.([]int32), b.([]int32)[0], out.([]int64), l)
	},
	{Int32, Int64, Float}: func(a, b, out any, l int) {
		ElemAdd(a.([]int32), b.([]int64)[0], out.([]float32), l)
	},
	{Int32, Int64, Double}: func(a, b, out any, l int) {
		ElemAdd(a.([]int32), b.([]int64)[0], out.([]float64), l)
	},
	{Int32, Int64, Int32}: func(a, b, out any, l int) {
		ElemAdd(a.([]int32), b.([]int64)[0], out.([]int32), l)
	},
	{Int32, Int64, Int64}: func(a, b, out any, l int) {
		ElemAdd(a.([]int32), b.([]int64)[0], out.([]int64), l)
	},
	{Int64, Float, Float}: func(a, b, out any, l int) {
		ElemAdd(a.([]int64), b.([]float32)[0], out.([]float32), l)
	},
	{Int64, Float, Double}: func(a, b, out any, l int) {
		ElemAdd(a.([]int64), b.([]float32)[0], out.([]float64), l)
	},
	{Int64, Float, Int32}: func(a, b, out any, l int) {
		ElemAdd(a.([]int64), b.([]float32)[0], out.([]int32), l)
	},
	{Int64, Float, Int64}: func(a, b, out any, l int) {
		ElemAdd(a.([]int64), b.([]float32)[0], out.([]int64), l)
	},
	{Int64, Double, Float}: func(a, b, out any, l int) {
		ElemAdd(a.([]int64), b.([]float64)[0], out.([]float32), l)
	},
	{Int64, Double, Double}: func(a, b, out any, l int) {
		ElemAdd(a.([]int64), b.([]float64)[0], out.([]float64), l)
	},
	{Int64, Double, Int32}: func(a, b, out any, l int) {
		ElemAdd(a.([]int64), b.([]float64)[0], out.([]int32), l)
	},
	{Int64, Double, Int64}: func(a, b, out any, l int) {
		ElemAdd(a.([]int64), b.([]float64)[0], out.([]int64), l)
	},
	{Int64, Int32, Float}: func(a, b, out any, l int) {
		ElemAdd(a.([]int64), b.([]int32)[0], out.([]float32), l)
	},
	{Int64, Int32, Double}: func(a, b, out any, l int) {
		ElemAdd(a.([]int64), b.([]int32)[0], out.([]float64), l)
	},
	{Int64, Int32, Int32}: func(a, b, out any, l int) {
		ElemAdd(a.([]int64), b.([]int32)[0], out.([]int32), l)
	},
	{Int64, Int32, Int64}: func(a, b, out any, l int) {
		ElemAdd(a.([]int64), b.([]int32)[0], out.([]int64), l)
	},
	{Int64, Int64, Float}: func(a, b, out any, l int) {
		ElemAdd(a.([]int64), b.([]int64)[0], out.([]float32), l)
	},
	{Int64, Int64, Double}: func(a, b, out any, l int) {
		ElemAdd(a.([]int64), b.([]int64)[0], out.([]float64), l)
	},
	{Int64, Int64, Int32}: func(a, b, out any, l int) {
		ElemAdd(a.([]int64), b.([]int64)[0], out.([]int32), l)
	},
	{Int64, Int64, Int64}: func(a, b, out any, l int) {
		ElemAdd(a.([]int64), b.([]int64)[0], out.([]int64), l)
	},
}

func ElemAdd[T Numeric, U Numeric, V Numeric](first []T, second U, result []V, length int) {
	elem := V(second)
	for i := range length {
		result[i] = V(first[i]) + elem
	}
}

func (t *Tensor) addElem(other *Tensor, out *Tensor) (*Tensor, error) {
	if out == nil {
		out = createOutputTensor(t.DType, other.DType, t.Shape)
	}
	length := t.Shape[0]
	if len(t.Shape) == 2 {
		length *= t.Shape[1]
	}

	key := addKey{t.DType, other.DType, out.DType}
	fn, ok := elemAddDispatch[key]
	if !ok {
		return nil, fmt.Errorf("unsupported add combination: %v + %v -> %v", t.DType, other.DType, out.DType)
	}
	fn(t.rawData(), other.rawData(), out.rawData(), length)
	return out, nil
}

type vecAddFunc func(a, b, out any, shape []int)

var rowAddDispatch = map[addKey]vecAddFunc{
	{Float, Float, Float}: func(a, b, out any, shape []int) {
		RowAdd(a.([]float32), b.([]float32), out.([]float32), shape)
	},
	{Float, Float, Double}: func(a, b, out any, shape []int) {
		RowAdd(a.([]float32), b.([]float32), out.([]float64), shape)
	},
	{Float, Float, Int32}: func(a, b, out any, shape []int) {
		RowAdd(a.([]float32), b.([]float32), out.([]int32), shape)
	},
	{Float, Float, Int64}: func(a, b, out any, shape []int) {
		RowAdd(a.([]float32), b.([]float32), out.([]int64), shape)
	},
	{Float, Double, Float}: func(a, b, out any, shape []int) {
		RowAdd(a.([]float32), b.([]float64), out.([]float32), shape)
	},
	{Float, Double, Double}: func(a, b, out any, shape []int) {
		RowAdd(a.([]float32), b.([]float64), out.([]float64), shape)
	},
	{Float, Double, Int32}: func(a, b, out any, shape []int) {
		RowAdd(a.([]float32), b.([]float64), out.([]int32), shape)
	},
	{Float, Double, Int64}: func(a, b, out any, shape []int) {
		RowAdd(a.([]float32), b.([]float64), out.([]int64), shape)
	},
	{Float, Int32, Float}: func(a, b, out any, shape []int) {
		RowAdd(a.([]float32), b.([]int32), out.([]float32), shape)
	},
	{Float, Int32, Double}: func(a, b, out any, shape []int) {
		RowAdd(a.([]float32), b.([]int32), out.([]float64), shape)
	},
	{Float, Int32, Int32}: func(a, b, out any, shape []int) {
		RowAdd(a.([]float32), b.([]int32), out.([]int32), shape)
	},
	{Float, Int32, Int64}: func(a, b, out any, shape []int) {
		RowAdd(a.([]float32), b.([]int32), out.([]int64), shape)
	},
	{Float, Int64, Float}: func(a, b, out any, shape []int) {
		RowAdd(a.([]float32), b.([]int64), out.([]float32), shape)
	},
	{Float, Int64, Double}: func(a, b, out any, shape []int) {
		RowAdd(a.([]float32), b.([]int64), out.([]float64), shape)
	},
	{Float, Int64, Int32}: func(a, b, out any, shape []int) {
		RowAdd(a.([]float32), b.([]int64), out.([]int32), shape)
	},
	{Float, Int64, Int64}: func(a, b, out any, shape []int) {
		RowAdd(a.([]float32), b.([]int64), out.([]int64), shape)
	},
	{Double, Float, Float}: func(a, b, out any, shape []int) {
		RowAdd(a.([]float64), b.([]float32), out.([]float32), shape)
	},
	{Double, Float, Double}: func(a, b, out any, shape []int) {
		RowAdd(a.([]float64), b.([]float32), out.([]float64), shape)
	},
	{Double, Float, Int32}: func(a, b, out any, shape []int) {
		RowAdd(a.([]float64), b.([]float32), out.([]int32), shape)
	},
	{Double, Float, Int64}: func(a, b, out any, shape []int) {
		RowAdd(a.([]float64), b.([]float32), out.([]int64), shape)
	},
	{Double, Double, Float}: func(a, b, out any, shape []int) {
		RowAdd(a.([]float64), b.([]float64), out.([]float32), shape)
	},
	{Double, Double, Double}: func(a, b, out any, shape []int) {
		RowAdd(a.([]float64), b.([]float64), out.([]float64), shape)
	},
	{Double, Double, Int32}: func(a, b, out any, shape []int) {
		RowAdd(a.([]float64), b.([]float64), out.([]int32), shape)
	},
	{Double, Double, Int64}: func(a, b, out any, shape []int) {
		RowAdd(a.([]float64), b.([]float64), out.([]int64), shape)
	},
	{Double, Int32, Float}: func(a, b, out any, shape []int) {
		RowAdd(a.([]float64), b.([]int32), out.([]float32), shape)
	},
	{Double, Int32, Double}: func(a, b, out any, shape []int) {
		RowAdd(a.([]float64), b.([]int32), out.([]float64), shape)
	},
	{Double, Int32, Int32}: func(a, b, out any, shape []int) {
		RowAdd(a.([]float64), b.([]int32), out.([]int32), shape)
	},
	{Double, Int32, Int64}: func(a, b, out any, shape []int) {
		RowAdd(a.([]float64), b.([]int32), out.([]int64), shape)
	},
	{Double, Int64, Float}: func(a, b, out any, shape []int) {
		RowAdd(a.([]float64), b.([]int64), out.([]float32), shape)
	},
	{Double, Int64, Double}: func(a, b, out any, shape []int) {
		RowAdd(a.([]float64), b.([]int64), out.([]float64), shape)
	},
	{Double, Int64, Int32}: func(a, b, out any, shape []int) {
		RowAdd(a.([]float64), b.([]int64), out.([]int32), shape)
	},
	{Double, Int64, Int64}: func(a, b, out any, shape []int) {
		RowAdd(a.([]float64), b.([]int64), out.([]int64), shape)
	},
	{Int32, Float, Float}: func(a, b, out any, shape []int) {
		RowAdd(a.([]int32), b.([]float32), out.([]float32), shape)
	},
	{Int32, Float, Double}: func(a, b, out any, shape []int) {
		RowAdd(a.([]int32), b.([]float32), out.([]float64), shape)
	},
	{Int32, Float, Int32}: func(a, b, out any, shape []int) {
		RowAdd(a.([]int32), b.([]float32), out.([]int32), shape)
	},
	{Int32, Float, Int64}: func(a, b, out any, shape []int) {
		RowAdd(a.([]int32), b.([]float32), out.([]int64), shape)
	},
	{Int32, Double, Float}: func(a, b, out any, shape []int) {
		RowAdd(a.([]int32), b.([]float64), out.([]float32), shape)
	},
	{Int32, Double, Double}: func(a, b, out any, shape []int) {
		RowAdd(a.([]int32), b.([]float64), out.([]float64), shape)
	},
	{Int32, Double, Int32}: func(a, b, out any, shape []int) {
		RowAdd(a.([]int32), b.([]float64), out.([]int32), shape)
	},
	{Int32, Double, Int64}: func(a, b, out any, shape []int) {
		RowAdd(a.([]int32), b.([]float64), out.([]int64), shape)
	},
	{Int32, Int32, Float}: func(a, b, out any, shape []int) {
		RowAdd(a.([]int32), b.([]int32), out.([]float32), shape)
	},
	{Int32, Int32, Double}: func(a, b, out any, shape []int) {
		RowAdd(a.([]int32), b.([]int32), out.([]float64), shape)
	},
	{Int32, Int32, Int32}: func(a, b, out any, shape []int) {
		RowAdd(a.([]int32), b.([]int32), out.([]int32), shape)
	},
	{Int32, Int32, Int64}: func(a, b, out any, shape []int) {
		RowAdd(a.([]int32), b.([]int32), out.([]int64), shape)
	},
	{Int32, Int64, Float}: func(a, b, out any, shape []int) {
		RowAdd(a.([]int32), b.([]int64), out.([]float32), shape)
	},
	{Int32, Int64, Double}: func(a, b, out any, shape []int) {
		RowAdd(a.([]int32), b.([]int64), out.([]float64), shape)
	},
	{Int32, Int64, Int32}: func(a, b, out any, shape []int) {
		RowAdd(a.([]int32), b.([]int64), out.([]int32), shape)
	},
	{Int32, Int64, Int64}: func(a, b, out any, shape []int) {
		RowAdd(a.([]int32), b.([]int64), out.([]int64), shape)
	},
	{Int64, Float, Float}: func(a, b, out any, shape []int) {
		RowAdd(a.([]int64), b.([]float32), out.([]float32), shape)
	},
	{Int64, Float, Double}: func(a, b, out any, shape []int) {
		RowAdd(a.([]int64), b.([]float32), out.([]float64), shape)
	},
	{Int64, Float, Int32}: func(a, b, out any, shape []int) {
		RowAdd(a.([]int64), b.([]float32), out.([]int32), shape)
	},
	{Int64, Float, Int64}: func(a, b, out any, shape []int) {
		RowAdd(a.([]int64), b.([]float32), out.([]int64), shape)
	},
	{Int64, Double, Float}: func(a, b, out any, shape []int) {
		RowAdd(a.([]int64), b.([]float64), out.([]float32), shape)
	},
	{Int64, Double, Double}: func(a, b, out any, shape []int) {
		RowAdd(a.([]int64), b.([]float64), out.([]float64), shape)
	},
	{Int64, Double, Int32}: func(a, b, out any, shape []int) {
		RowAdd(a.([]int64), b.([]float64), out.([]int32), shape)
	},
	{Int64, Double, Int64}: func(a, b, out any, shape []int) {
		RowAdd(a.([]int64), b.([]float64), out.([]int64), shape)
	},
	{Int64, Int32, Float}: func(a, b, out any, shape []int) {
		RowAdd(a.([]int64), b.([]int32), out.([]float32), shape)
	},
	{Int64, Int32, Double}: func(a, b, out any, shape []int) {
		RowAdd(a.([]int64), b.([]int32), out.([]float64), shape)
	},
	{Int64, Int32, Int32}: func(a, b, out any, shape []int) {
		RowAdd(a.([]int64), b.([]int32), out.([]int32), shape)
	},
	{Int64, Int32, Int64}: func(a, b, out any, shape []int) {
		RowAdd(a.([]int64), b.([]int32), out.([]int64), shape)
	},
	{Int64, Int64, Float}: func(a, b, out any, shape []int) {
		RowAdd(a.([]int64), b.([]int64), out.([]float32), shape)
	},
	{Int64, Int64, Double}: func(a, b, out any, shape []int) {
		RowAdd(a.([]int64), b.([]int64), out.([]float64), shape)
	},
	{Int64, Int64, Int32}: func(a, b, out any, shape []int) {
		RowAdd(a.([]int64), b.([]int64), out.([]int32), shape)
	},
	{Int64, Int64, Int64}: func(a, b, out any, shape []int) {
		RowAdd(a.([]int64), b.([]int64), out.([]int64), shape)
	},
}

func RowAdd[T Numeric, U Numeric, V Numeric](first []T, row []U, result []V, shape []int) {
	for i := range shape[0] {
		for j := range shape[1] {
			result[i*shape[1]+j] = V(first[i*shape[1]+j]) + V(row[j])
		}
	}
}

func (t *Tensor) addRow(other *Tensor, out *Tensor) (*Tensor, error) {
	rowLength := other.Shape[0]
	if len(other.Shape) == 2 {
		rowLength *= other.Shape[1]
	}
	if rowLength != t.Shape[1] {
		return nil, fmt.Errorf("operants rows don't align: %d != %d", rowLength, t.Shape[1])
	}
	if out == nil {
		out = createOutputTensor(t.DType, other.DType, t.Shape)
	}
	key := addKey{t.DType, other.DType, out.DType}
	fn, ok := rowAddDispatch[key]
	if !ok {
		return nil, fmt.Errorf("unsupported add combination: %v + %v -> %v", t.DType, other.DType, out.DType)
	}
	fn(t.rawData(), other.rawData(), out.rawData(), t.Shape)
	return out, nil
}

var colAddDispatch = map[addKey]vecAddFunc{
	{Float, Float, Float}: func(a, b, out any, shape []int) {
		ColAdd(a.([]float32), b.([]float32), out.([]float32), shape)
	},
	{Float, Float, Double}: func(a, b, out any, shape []int) {
		ColAdd(a.([]float32), b.([]float32), out.([]float64), shape)
	},
	{Float, Float, Int32}: func(a, b, out any, shape []int) {
		ColAdd(a.([]float32), b.([]float32), out.([]int32), shape)
	},
	{Float, Float, Int64}: func(a, b, out any, shape []int) {
		ColAdd(a.([]float32), b.([]float32), out.([]int64), shape)
	},
	{Float, Double, Float}: func(a, b, out any, shape []int) {
		ColAdd(a.([]float32), b.([]float64), out.([]float32), shape)
	},
	{Float, Double, Double}: func(a, b, out any, shape []int) {
		ColAdd(a.([]float32), b.([]float64), out.([]float64), shape)
	},
	{Float, Double, Int32}: func(a, b, out any, shape []int) {
		ColAdd(a.([]float32), b.([]float64), out.([]int32), shape)
	},
	{Float, Double, Int64}: func(a, b, out any, shape []int) {
		ColAdd(a.([]float32), b.([]float64), out.([]int64), shape)
	},
	{Float, Int32, Float}: func(a, b, out any, shape []int) {
		ColAdd(a.([]float32), b.([]int32), out.([]float32), shape)
	},
	{Float, Int32, Double}: func(a, b, out any, shape []int) {
		ColAdd(a.([]float32), b.([]int32), out.([]float64), shape)
	},
	{Float, Int32, Int32}: func(a, b, out any, shape []int) {
		ColAdd(a.([]float32), b.([]int32), out.([]int32), shape)
	},
	{Float, Int32, Int64}: func(a, b, out any, shape []int) {
		ColAdd(a.([]float32), b.([]int32), out.([]int64), shape)
	},
	{Float, Int64, Float}: func(a, b, out any, shape []int) {
		ColAdd(a.([]float32), b.([]int64), out.([]float32), shape)
	},
	{Float, Int64, Double}: func(a, b, out any, shape []int) {
		ColAdd(a.([]float32), b.([]int64), out.([]float64), shape)
	},
	{Float, Int64, Int32}: func(a, b, out any, shape []int) {
		ColAdd(a.([]float32), b.([]int64), out.([]int32), shape)
	},
	{Float, Int64, Int64}: func(a, b, out any, shape []int) {
		ColAdd(a.([]float32), b.([]int64), out.([]int64), shape)
	},
	{Double, Float, Float}: func(a, b, out any, shape []int) {
		ColAdd(a.([]float64), b.([]float32), out.([]float32), shape)
	},
	{Double, Float, Double}: func(a, b, out any, shape []int) {
		ColAdd(a.([]float64), b.([]float32), out.([]float64), shape)
	},
	{Double, Float, Int32}: func(a, b, out any, shape []int) {
		ColAdd(a.([]float64), b.([]float32), out.([]int32), shape)
	},
	{Double, Float, Int64}: func(a, b, out any, shape []int) {
		ColAdd(a.([]float64), b.([]float32), out.([]int64), shape)
	},
	{Double, Double, Float}: func(a, b, out any, shape []int) {
		ColAdd(a.([]float64), b.([]float64), out.([]float32), shape)
	},
	{Double, Double, Double}: func(a, b, out any, shape []int) {
		ColAdd(a.([]float64), b.([]float64), out.([]float64), shape)
	},
	{Double, Double, Int32}: func(a, b, out any, shape []int) {
		ColAdd(a.([]float64), b.([]float64), out.([]int32), shape)
	},
	{Double, Double, Int64}: func(a, b, out any, shape []int) {
		ColAdd(a.([]float64), b.([]float64), out.([]int64), shape)
	},
	{Double, Int32, Float}: func(a, b, out any, shape []int) {
		ColAdd(a.([]float64), b.([]int32), out.([]float32), shape)
	},
	{Double, Int32, Double}: func(a, b, out any, shape []int) {
		ColAdd(a.([]float64), b.([]int32), out.([]float64), shape)
	},
	{Double, Int32, Int32}: func(a, b, out any, shape []int) {
		ColAdd(a.([]float64), b.([]int32), out.([]int32), shape)
	},
	{Double, Int32, Int64}: func(a, b, out any, shape []int) {
		ColAdd(a.([]float64), b.([]int32), out.([]int64), shape)
	},
	{Double, Int64, Float}: func(a, b, out any, shape []int) {
		ColAdd(a.([]float64), b.([]int64), out.([]float32), shape)
	},
	{Double, Int64, Double}: func(a, b, out any, shape []int) {
		ColAdd(a.([]float64), b.([]int64), out.([]float64), shape)
	},
	{Double, Int64, Int32}: func(a, b, out any, shape []int) {
		ColAdd(a.([]float64), b.([]int64), out.([]int32), shape)
	},
	{Double, Int64, Int64}: func(a, b, out any, shape []int) {
		ColAdd(a.([]float64), b.([]int64), out.([]int64), shape)
	},
	{Int32, Float, Float}: func(a, b, out any, shape []int) {
		ColAdd(a.([]int32), b.([]float32), out.([]float32), shape)
	},
	{Int32, Float, Double}: func(a, b, out any, shape []int) {
		ColAdd(a.([]int32), b.([]float32), out.([]float64), shape)
	},
	{Int32, Float, Int32}: func(a, b, out any, shape []int) {
		ColAdd(a.([]int32), b.([]float32), out.([]int32), shape)
	},
	{Int32, Float, Int64}: func(a, b, out any, shape []int) {
		ColAdd(a.([]int32), b.([]float32), out.([]int64), shape)
	},
	{Int32, Double, Float}: func(a, b, out any, shape []int) {
		ColAdd(a.([]int32), b.([]float64), out.([]float32), shape)
	},
	{Int32, Double, Double}: func(a, b, out any, shape []int) {
		ColAdd(a.([]int32), b.([]float64), out.([]float64), shape)
	},
	{Int32, Double, Int32}: func(a, b, out any, shape []int) {
		ColAdd(a.([]int32), b.([]float64), out.([]int32), shape)
	},
	{Int32, Double, Int64}: func(a, b, out any, shape []int) {
		ColAdd(a.([]int32), b.([]float64), out.([]int64), shape)
	},
	{Int32, Int32, Float}: func(a, b, out any, shape []int) {
		ColAdd(a.([]int32), b.([]int32), out.([]float32), shape)
	},
	{Int32, Int32, Double}: func(a, b, out any, shape []int) {
		ColAdd(a.([]int32), b.([]int32), out.([]float64), shape)
	},
	{Int32, Int32, Int32}: func(a, b, out any, shape []int) {
		ColAdd(a.([]int32), b.([]int32), out.([]int32), shape)
	},
	{Int32, Int32, Int64}: func(a, b, out any, shape []int) {
		ColAdd(a.([]int32), b.([]int32), out.([]int64), shape)
	},
	{Int32, Int64, Float}: func(a, b, out any, shape []int) {
		ColAdd(a.([]int32), b.([]int64), out.([]float32), shape)
	},
	{Int32, Int64, Double}: func(a, b, out any, shape []int) {
		ColAdd(a.([]int32), b.([]int64), out.([]float64), shape)
	},
	{Int32, Int64, Int32}: func(a, b, out any, shape []int) {
		ColAdd(a.([]int32), b.([]int64), out.([]int32), shape)
	},
	{Int32, Int64, Int64}: func(a, b, out any, shape []int) {
		ColAdd(a.([]int32), b.([]int64), out.([]int64), shape)
	},
	{Int64, Float, Float}: func(a, b, out any, shape []int) {
		ColAdd(a.([]int64), b.([]float32), out.([]float32), shape)
	},
	{Int64, Float, Double}: func(a, b, out any, shape []int) {
		ColAdd(a.([]int64), b.([]float32), out.([]float64), shape)
	},
	{Int64, Float, Int32}: func(a, b, out any, shape []int) {
		ColAdd(a.([]int64), b.([]float32), out.([]int32), shape)
	},
	{Int64, Float, Int64}: func(a, b, out any, shape []int) {
		ColAdd(a.([]int64), b.([]float32), out.([]int64), shape)
	},
	{Int64, Double, Float}: func(a, b, out any, shape []int) {
		ColAdd(a.([]int64), b.([]float64), out.([]float32), shape)
	},
	{Int64, Double, Double}: func(a, b, out any, shape []int) {
		ColAdd(a.([]int64), b.([]float64), out.([]float64), shape)
	},
	{Int64, Double, Int32}: func(a, b, out any, shape []int) {
		ColAdd(a.([]int64), b.([]float64), out.([]int32), shape)
	},
	{Int64, Double, Int64}: func(a, b, out any, shape []int) {
		ColAdd(a.([]int64), b.([]float64), out.([]int64), shape)
	},
	{Int64, Int32, Float}: func(a, b, out any, shape []int) {
		ColAdd(a.([]int64), b.([]int32), out.([]float32), shape)
	},
	{Int64, Int32, Double}: func(a, b, out any, shape []int) {
		ColAdd(a.([]int64), b.([]int32), out.([]float64), shape)
	},
	{Int64, Int32, Int32}: func(a, b, out any, shape []int) {
		ColAdd(a.([]int64), b.([]int32), out.([]int32), shape)
	},
	{Int64, Int32, Int64}: func(a, b, out any, shape []int) {
		ColAdd(a.([]int64), b.([]int32), out.([]int64), shape)
	},
	{Int64, Int64, Float}: func(a, b, out any, shape []int) {
		ColAdd(a.([]int64), b.([]int64), out.([]float32), shape)
	},
	{Int64, Int64, Double}: func(a, b, out any, shape []int) {
		ColAdd(a.([]int64), b.([]int64), out.([]float64), shape)
	},
	{Int64, Int64, Int32}: func(a, b, out any, shape []int) {
		ColAdd(a.([]int64), b.([]int64), out.([]int32), shape)
	},
	{Int64, Int64, Int64}: func(a, b, out any, shape []int) {
		ColAdd(a.([]int64), b.([]int64), out.([]int64), shape)
	},
}

func ColAdd[T Numeric, U Numeric, V Numeric](first []T, col []U, result []V, shape []int) {
	for i := range shape[0] {
		elem := V(col[i])
		for j := range shape[1] {
			result[i*shape[1]+j] = V(first[i*shape[1]+j]) + elem
		}
	}
}

func (t *Tensor) addCol(other *Tensor, out *Tensor) (*Tensor, error) {
	colLength := other.Shape[0]
	if colLength != t.Shape[0] {
		return nil, fmt.Errorf("operants cols don't align: %d != %d", colLength, t.Shape[0])
	}
	if out == nil {
		out = createOutputTensor(t.DType, other.DType, t.Shape)
	}

	key := addKey{t.DType, other.DType, out.DType}
	fn, ok := colAddDispatch[key]
	if !ok {
		return nil, fmt.Errorf("unsupported add combination: %v + %v -> %v", t.DType, other.DType, out.DType)
	}
	fn(t.rawData(), other.rawData(), out.rawData(), t.Shape)
	return out, nil
}
