package tensor

import (
	"errors"
	"fmt"
)

type Numeric interface {
	int32 | int64 | float32 | float64
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

func OpAdd[T Numeric, U Numeric, V Numeric](first []T, second []U, result []V, length int) {
	for i := range length {
		result[i] = V(first[i]) + V(second[i])
	}
}

func ElemAdd[T Numeric, U Numeric, V Numeric](first []T, second U, result []V, length int) {
	elem := V(second)
	for i := range length {
		result[i] = V(first[i]) + elem
	}
}

func RowAdd[T Numeric, U Numeric, V Numeric](first []T, row []U, result []V, shape []int) {
	for i := range shape[0] {
		for j := range shape[1] {
			result[i*shape[1]+j] = V(first[i*shape[1]+j]) + V(row[j])
		}
	}
}

func ColAdd[T Numeric, U Numeric, V Numeric](first []T, col []U, result []V, shape []int) {
	for i := range shape[0] {
		elem := V(col[i])
		for j := range shape[1] {
			result[i*shape[1]+j] = V(first[i*shape[1]+j]) + elem
		}
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
	if t.DType == Float && other.DType == Float {
		switch out.DType {
		case Float:
			OpAdd(t.FloatData, other.FloatData, out.FloatData, length)
		case Double:
			OpAdd(t.FloatData, other.FloatData, out.DoubleData, length)
		case Int32:
			OpAdd(t.FloatData, other.FloatData, out.Int32Data, length)
		case Int64:
			OpAdd(t.FloatData, other.FloatData, out.Int64Data, length)
		}
	} else if t.DType == Float && other.DType == Double {
		switch out.DType {
		case Float:
			OpAdd(t.FloatData, other.DoubleData, out.FloatData, length)
		case Double:
			OpAdd(t.FloatData, other.DoubleData, out.DoubleData, length)
		case Int32:
			OpAdd(t.FloatData, other.DoubleData, out.Int32Data, length)
		case Int64:
			OpAdd(t.FloatData, other.DoubleData, out.Int64Data, length)
		}
	} else if t.DType == Float && other.DType == Int32 {
		switch out.DType {
		case Float:
			OpAdd(t.FloatData, other.Int32Data, out.FloatData, length)
		case Double:
			OpAdd(t.FloatData, other.Int32Data, out.DoubleData, length)
		case Int32:
			OpAdd(t.FloatData, other.Int32Data, out.Int32Data, length)
		case Int64:
			OpAdd(t.FloatData, other.Int32Data, out.Int64Data, length)
		}
	} else if t.DType == Float && other.DType == Int64 {
		switch out.DType {
		case Float:
			OpAdd(t.FloatData, other.Int64Data, out.FloatData, length)
		case Double:
			OpAdd(t.FloatData, other.Int64Data, out.DoubleData, length)
		case Int32:
			OpAdd(t.FloatData, other.Int64Data, out.Int32Data, length)
		case Int64:
			OpAdd(t.FloatData, other.Int64Data, out.Int64Data, length)
		}
	} else if t.DType == Double && other.DType == Float {
		switch out.DType {
		case Float:
			OpAdd(t.DoubleData, other.FloatData, out.FloatData, length)
		case Double:
			OpAdd(t.DoubleData, other.FloatData, out.DoubleData, length)
		case Int32:
			OpAdd(t.DoubleData, other.FloatData, out.Int32Data, length)
		case Int64:
			OpAdd(t.DoubleData, other.FloatData, out.Int64Data, length)
		}
	} else if t.DType == Double && other.DType == Double {
		switch out.DType {
		case Float:
			OpAdd(t.DoubleData, other.DoubleData, out.FloatData, length)
		case Double:
			OpAdd(t.DoubleData, other.DoubleData, out.DoubleData, length)
		case Int32:
			OpAdd(t.DoubleData, other.DoubleData, out.Int32Data, length)
		case Int64:
			OpAdd(t.DoubleData, other.DoubleData, out.Int64Data, length)
		}
	} else if t.DType == Double && other.DType == Int32 {
		switch out.DType {
		case Float:
			OpAdd(t.DoubleData, other.Int32Data, out.FloatData, length)
		case Double:
			OpAdd(t.DoubleData, other.Int32Data, out.DoubleData, length)
		case Int32:
			OpAdd(t.DoubleData, other.Int32Data, out.Int32Data, length)
		case Int64:
			OpAdd(t.DoubleData, other.Int32Data, out.Int64Data, length)
		}
	} else if t.DType == Double && other.DType == Int64 {
		switch out.DType {
		case Float:
			OpAdd(t.DoubleData, other.Int64Data, out.FloatData, length)
		case Double:
			OpAdd(t.DoubleData, other.Int64Data, out.DoubleData, length)
		case Int32:
			OpAdd(t.DoubleData, other.Int64Data, out.Int32Data, length)
		case Int64:
			OpAdd(t.DoubleData, other.Int64Data, out.Int64Data, length)
		}
	} else if t.DType == Int32 && other.DType == Float {
		switch out.DType {
		case Float:
			OpAdd(t.Int32Data, other.FloatData, out.FloatData, length)
		case Double:
			OpAdd(t.Int32Data, other.FloatData, out.DoubleData, length)
		case Int32:
			OpAdd(t.Int32Data, other.FloatData, out.Int32Data, length)
		case Int64:
			OpAdd(t.Int32Data, other.FloatData, out.Int64Data, length)
		}
	} else if t.DType == Int32 && other.DType == Double {
		switch out.DType {
		case Float:
			OpAdd(t.Int32Data, other.DoubleData, out.FloatData, length)
		case Double:
			OpAdd(t.Int32Data, other.DoubleData, out.DoubleData, length)
		case Int32:
			OpAdd(t.Int32Data, other.DoubleData, out.Int32Data, length)
		case Int64:
			OpAdd(t.Int32Data, other.DoubleData, out.Int64Data, length)
		}
	} else if t.DType == Int32 && other.DType == Int32 {
		switch out.DType {
		case Float:
			OpAdd(t.Int32Data, other.Int32Data, out.FloatData, length)
		case Double:
			OpAdd(t.Int32Data, other.Int32Data, out.DoubleData, length)
		case Int32:
			OpAdd(t.Int32Data, other.Int32Data, out.Int32Data, length)
		case Int64:
			OpAdd(t.Int32Data, other.Int32Data, out.Int64Data, length)
		}
	} else if t.DType == Int32 && other.DType == Int64 {
		switch out.DType {
		case Float:
			OpAdd(t.Int32Data, other.Int64Data, out.FloatData, length)
		case Double:
			OpAdd(t.Int32Data, other.Int64Data, out.DoubleData, length)
		case Int32:
			OpAdd(t.Int32Data, other.Int64Data, out.Int32Data, length)
		case Int64:
			OpAdd(t.Int32Data, other.Int64Data, out.Int64Data, length)
		}
	} else if t.DType == Int64 && other.DType == Float {
		switch out.DType {
		case Float:
			OpAdd(t.Int64Data, other.FloatData, out.FloatData, length)
		case Double:
			OpAdd(t.Int64Data, other.FloatData, out.DoubleData, length)
		case Int32:
			OpAdd(t.Int64Data, other.FloatData, out.Int32Data, length)
		case Int64:
			OpAdd(t.Int64Data, other.FloatData, out.Int64Data, length)
		}
	} else if t.DType == Int64 && other.DType == Double {
		switch out.DType {
		case Float:
			OpAdd(t.Int64Data, other.DoubleData, out.FloatData, length)
		case Double:
			OpAdd(t.Int64Data, other.DoubleData, out.DoubleData, length)
		case Int32:
			OpAdd(t.Int64Data, other.DoubleData, out.Int32Data, length)
		case Int64:
			OpAdd(t.Int64Data, other.DoubleData, out.Int64Data, length)
		}
	} else if t.DType == Int64 && other.DType == Int32 {
		switch out.DType {
		case Float:
			OpAdd(t.Int64Data, other.Int32Data, out.FloatData, length)
		case Double:
			OpAdd(t.Int64Data, other.Int32Data, out.DoubleData, length)
		case Int32:
			OpAdd(t.Int64Data, other.Int32Data, out.Int32Data, length)
		case Int64:
			OpAdd(t.Int64Data, other.Int32Data, out.Int64Data, length)
		}
	} else if t.DType == Int64 && other.DType == Int64 {
		switch out.DType {
		case Float:
			OpAdd(t.Int64Data, other.Int64Data, out.FloatData, length)
		case Double:
			OpAdd(t.Int64Data, other.Int64Data, out.DoubleData, length)
		case Int32:
			OpAdd(t.Int64Data, other.Int64Data, out.Int32Data, length)
		case Int64:
			OpAdd(t.Int64Data, other.Int64Data, out.Int64Data, length)
		}
	}
	return out, nil
}

func (t *Tensor) addElem(other *Tensor, out *Tensor) (*Tensor, error) {
	if out == nil {
		out = createOutputTensor(t.DType, other.DType, t.Shape)
	}
	length := t.Shape[0]
	if len(t.Shape) == 2 {
		length *= t.Shape[1]
	}
	if t.DType == Float && other.DType == Float {
		switch out.DType {
		case Float:
			ElemAdd(t.FloatData, other.FloatData[0], out.FloatData, length)
		case Double:
			ElemAdd(t.FloatData, other.FloatData[0], out.DoubleData, length)
		case Int32:
			ElemAdd(t.FloatData, other.FloatData[0], out.Int32Data, length)
		case Int64:
			ElemAdd(t.FloatData, other.FloatData[0], out.Int64Data, length)
		}
	} else if t.DType == Float && other.DType == Double {
		switch out.DType {
		case Float:
			ElemAdd(t.FloatData, other.DoubleData[0], out.FloatData, length)
		case Double:
			ElemAdd(t.FloatData, other.DoubleData[0], out.DoubleData, length)
		case Int32:
			ElemAdd(t.FloatData, other.DoubleData[0], out.Int32Data, length)
		case Int64:
			ElemAdd(t.FloatData, other.DoubleData[0], out.Int64Data, length)
		}
	} else if t.DType == Float && other.DType == Int32 {
		switch out.DType {
		case Float:
			ElemAdd(t.FloatData, other.Int32Data[0], out.FloatData, length)
		case Double:
			ElemAdd(t.FloatData, other.Int32Data[0], out.DoubleData, length)
		case Int32:
			ElemAdd(t.FloatData, other.Int32Data[0], out.Int32Data, length)
		case Int64:
			ElemAdd(t.FloatData, other.Int32Data[0], out.Int64Data, length)
		}
	} else if t.DType == Float && other.DType == Int64 {
		switch out.DType {
		case Float:
			ElemAdd(t.FloatData, other.Int64Data[0], out.FloatData, length)
		case Double:
			ElemAdd(t.FloatData, other.Int64Data[0], out.DoubleData, length)
		case Int32:
			ElemAdd(t.FloatData, other.Int64Data[0], out.Int32Data, length)
		case Int64:
			ElemAdd(t.FloatData, other.Int64Data[0], out.Int64Data, length)
		}
	} else if t.DType == Double && other.DType == Float {
		switch out.DType {
		case Float:
			ElemAdd(t.DoubleData, other.FloatData[0], out.FloatData, length)
		case Double:
			ElemAdd(t.DoubleData, other.FloatData[0], out.DoubleData, length)
		case Int32:
			ElemAdd(t.DoubleData, other.FloatData[0], out.Int32Data, length)
		case Int64:
			ElemAdd(t.DoubleData, other.FloatData[0], out.Int64Data, length)
		}
	} else if t.DType == Double && other.DType == Double {
		switch out.DType {
		case Float:
			ElemAdd(t.DoubleData, other.DoubleData[0], out.FloatData, length)
		case Double:
			ElemAdd(t.DoubleData, other.DoubleData[0], out.DoubleData, length)
		case Int32:
			ElemAdd(t.DoubleData, other.DoubleData[0], out.Int32Data, length)
		case Int64:
			ElemAdd(t.DoubleData, other.DoubleData[0], out.Int64Data, length)
		}
	} else if t.DType == Double && other.DType == Int32 {
		switch out.DType {
		case Float:
			ElemAdd(t.DoubleData, other.Int32Data[0], out.FloatData, length)
		case Double:
			ElemAdd(t.DoubleData, other.Int32Data[0], out.DoubleData, length)
		case Int32:
			ElemAdd(t.DoubleData, other.Int32Data[0], out.Int32Data, length)
		case Int64:
			ElemAdd(t.DoubleData, other.Int32Data[0], out.Int64Data, length)
		}
	} else if t.DType == Double && other.DType == Int64 {
		switch out.DType {
		case Float:
			ElemAdd(t.DoubleData, other.Int64Data[0], out.FloatData, length)
		case Double:
			ElemAdd(t.DoubleData, other.Int64Data[0], out.DoubleData, length)
		case Int32:
			ElemAdd(t.DoubleData, other.Int64Data[0], out.Int32Data, length)
		case Int64:
			ElemAdd(t.DoubleData, other.Int64Data[0], out.Int64Data, length)
		}
	} else if t.DType == Int32 && other.DType == Float {
		switch out.DType {
		case Float:
			ElemAdd(t.Int32Data, other.FloatData[0], out.FloatData, length)
		case Double:
			ElemAdd(t.Int32Data, other.FloatData[0], out.DoubleData, length)
		case Int32:
			ElemAdd(t.Int32Data, other.FloatData[0], out.Int32Data, length)
		case Int64:
			ElemAdd(t.Int32Data, other.FloatData[0], out.Int64Data, length)
		}
	} else if t.DType == Int32 && other.DType == Double {
		switch out.DType {
		case Float:
			ElemAdd(t.Int32Data, other.DoubleData[0], out.FloatData, length)
		case Double:
			ElemAdd(t.Int32Data, other.DoubleData[0], out.DoubleData, length)
		case Int32:
			ElemAdd(t.Int32Data, other.DoubleData[0], out.Int32Data, length)
		case Int64:
			ElemAdd(t.Int32Data, other.DoubleData[0], out.Int64Data, length)
		}
	} else if t.DType == Int32 && other.DType == Int32 {
		switch out.DType {
		case Float:
			ElemAdd(t.Int32Data, other.Int32Data[0], out.FloatData, length)
		case Double:
			ElemAdd(t.Int32Data, other.Int32Data[0], out.DoubleData, length)
		case Int32:
			ElemAdd(t.Int32Data, other.Int32Data[0], out.Int32Data, length)
		case Int64:
			ElemAdd(t.Int32Data, other.Int32Data[0], out.Int64Data, length)
		}
	} else if t.DType == Int32 && other.DType == Int64 {
		switch out.DType {
		case Float:
			ElemAdd(t.Int32Data, other.Int64Data[0], out.FloatData, length)
		case Double:
			ElemAdd(t.Int32Data, other.Int64Data[0], out.DoubleData, length)
		case Int32:
			ElemAdd(t.Int32Data, other.Int64Data[0], out.Int32Data, length)
		case Int64:
			ElemAdd(t.Int32Data, other.Int64Data[0], out.Int64Data, length)
		}
	} else if t.DType == Int64 && other.DType == Float {
		switch out.DType {
		case Float:
			ElemAdd(t.Int64Data, other.FloatData[0], out.FloatData, length)
		case Double:
			ElemAdd(t.Int64Data, other.FloatData[0], out.DoubleData, length)
		case Int32:
			ElemAdd(t.Int64Data, other.FloatData[0], out.Int32Data, length)
		case Int64:
			ElemAdd(t.Int64Data, other.FloatData[0], out.Int64Data, length)
		}
	} else if t.DType == Int64 && other.DType == Double {
		switch out.DType {
		case Float:
			ElemAdd(t.Int64Data, other.DoubleData[0], out.FloatData, length)
		case Double:
			ElemAdd(t.Int64Data, other.DoubleData[0], out.DoubleData, length)
		case Int32:
			ElemAdd(t.Int64Data, other.DoubleData[0], out.Int32Data, length)
		case Int64:
			ElemAdd(t.Int64Data, other.DoubleData[0], out.Int64Data, length)
		}
	} else if t.DType == Int64 && other.DType == Int32 {
		switch out.DType {
		case Float:
			ElemAdd(t.Int64Data, other.Int32Data[0], out.FloatData, length)
		case Double:
			ElemAdd(t.Int64Data, other.Int32Data[0], out.DoubleData, length)
		case Int32:
			ElemAdd(t.Int64Data, other.Int32Data[0], out.Int32Data, length)
		case Int64:
			ElemAdd(t.Int64Data, other.Int32Data[0], out.Int64Data, length)
		}
	} else if t.DType == Int64 && other.DType == Int64 {
		switch out.DType {
		case Float:
			ElemAdd(t.Int64Data, other.Int64Data[0], out.FloatData, length)
		case Double:
			ElemAdd(t.Int64Data, other.Int64Data[0], out.DoubleData, length)
		case Int32:
			ElemAdd(t.Int64Data, other.Int64Data[0], out.Int32Data, length)
		case Int64:
			ElemAdd(t.Int64Data, other.Int64Data[0], out.Int64Data, length)
		}
	}
	return out, nil
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
	if t.DType == Float && other.DType == Float {
		switch out.DType {
		case Float:
			RowAdd(t.FloatData, other.FloatData, out.FloatData, t.Shape)
		case Double:
			RowAdd(t.FloatData, other.FloatData, out.DoubleData, t.Shape)
		case Int32:
			RowAdd(t.FloatData, other.FloatData, out.Int32Data, t.Shape)
		case Int64:
			RowAdd(t.FloatData, other.FloatData, out.Int64Data, t.Shape)
		}
	} else if t.DType == Float && other.DType == Double {
		switch out.DType {
		case Float:
			RowAdd(t.FloatData, other.DoubleData, out.FloatData, t.Shape)
		case Double:
			RowAdd(t.FloatData, other.DoubleData, out.DoubleData, t.Shape)
		case Int32:
			RowAdd(t.FloatData, other.DoubleData, out.Int32Data, t.Shape)
		case Int64:
			RowAdd(t.FloatData, other.DoubleData, out.Int64Data, t.Shape)
		}
	} else if t.DType == Float && other.DType == Int32 {
		switch out.DType {
		case Float:
			RowAdd(t.FloatData, other.Int32Data, out.FloatData, t.Shape)
		case Double:
			RowAdd(t.FloatData, other.Int32Data, out.DoubleData, t.Shape)
		case Int32:
			RowAdd(t.FloatData, other.Int32Data, out.Int32Data, t.Shape)
		case Int64:
			RowAdd(t.FloatData, other.Int32Data, out.Int64Data, t.Shape)
		}
	} else if t.DType == Float && other.DType == Int64 {
		switch out.DType {
		case Float:
			RowAdd(t.FloatData, other.Int64Data, out.FloatData, t.Shape)
		case Double:
			RowAdd(t.FloatData, other.Int64Data, out.DoubleData, t.Shape)
		case Int32:
			RowAdd(t.FloatData, other.Int64Data, out.Int32Data, t.Shape)
		case Int64:
			RowAdd(t.FloatData, other.Int64Data, out.Int64Data, t.Shape)
		}
	} else if t.DType == Double && other.DType == Float {
		switch out.DType {
		case Float:
			RowAdd(t.DoubleData, other.FloatData, out.FloatData, t.Shape)
		case Double:
			RowAdd(t.DoubleData, other.FloatData, out.DoubleData, t.Shape)
		case Int32:
			RowAdd(t.DoubleData, other.FloatData, out.Int32Data, t.Shape)
		case Int64:
			RowAdd(t.DoubleData, other.FloatData, out.Int64Data, t.Shape)
		}
	} else if t.DType == Double && other.DType == Double {
		switch out.DType {
		case Float:
			RowAdd(t.DoubleData, other.DoubleData, out.FloatData, t.Shape)
		case Double:
			RowAdd(t.DoubleData, other.DoubleData, out.DoubleData, t.Shape)
		case Int32:
			RowAdd(t.DoubleData, other.DoubleData, out.Int32Data, t.Shape)
		case Int64:
			RowAdd(t.DoubleData, other.DoubleData, out.Int64Data, t.Shape)
		}
	} else if t.DType == Double && other.DType == Int32 {
		switch out.DType {
		case Float:
			RowAdd(t.DoubleData, other.Int32Data, out.FloatData, t.Shape)
		case Double:
			RowAdd(t.DoubleData, other.Int32Data, out.DoubleData, t.Shape)
		case Int32:
			RowAdd(t.DoubleData, other.Int32Data, out.Int32Data, t.Shape)
		case Int64:
			RowAdd(t.DoubleData, other.Int32Data, out.Int64Data, t.Shape)
		}
	} else if t.DType == Double && other.DType == Int64 {
		switch out.DType {
		case Float:
			RowAdd(t.DoubleData, other.Int64Data, out.FloatData, t.Shape)
		case Double:
			RowAdd(t.DoubleData, other.Int64Data, out.DoubleData, t.Shape)
		case Int32:
			RowAdd(t.DoubleData, other.Int64Data, out.Int32Data, t.Shape)
		case Int64:
			RowAdd(t.DoubleData, other.Int64Data, out.Int64Data, t.Shape)
		}
	} else if t.DType == Int32 && other.DType == Float {
		switch out.DType {
		case Float:
			RowAdd(t.Int32Data, other.FloatData, out.FloatData, t.Shape)
		case Double:
			RowAdd(t.Int32Data, other.FloatData, out.DoubleData, t.Shape)
		case Int32:
			RowAdd(t.Int32Data, other.FloatData, out.Int32Data, t.Shape)
		case Int64:
			RowAdd(t.Int32Data, other.FloatData, out.Int64Data, t.Shape)
		}
	} else if t.DType == Int32 && other.DType == Double {
		switch out.DType {
		case Float:
			RowAdd(t.Int32Data, other.DoubleData, out.FloatData, t.Shape)
		case Double:
			RowAdd(t.Int32Data, other.DoubleData, out.DoubleData, t.Shape)
		case Int32:
			RowAdd(t.Int32Data, other.DoubleData, out.Int32Data, t.Shape)
		case Int64:
			RowAdd(t.Int32Data, other.DoubleData, out.Int64Data, t.Shape)
		}
	} else if t.DType == Int32 && other.DType == Int32 {
		switch out.DType {
		case Float:
			RowAdd(t.Int32Data, other.Int32Data, out.FloatData, t.Shape)
		case Double:
			RowAdd(t.Int32Data, other.Int32Data, out.DoubleData, t.Shape)
		case Int32:
			RowAdd(t.Int32Data, other.Int32Data, out.Int32Data, t.Shape)
		case Int64:
			RowAdd(t.Int32Data, other.Int32Data, out.Int64Data, t.Shape)
		}
	} else if t.DType == Int32 && other.DType == Int64 {
		switch out.DType {
		case Float:
			RowAdd(t.Int32Data, other.Int64Data, out.FloatData, t.Shape)
		case Double:
			RowAdd(t.Int32Data, other.Int64Data, out.DoubleData, t.Shape)
		case Int32:
			RowAdd(t.Int32Data, other.Int64Data, out.Int32Data, t.Shape)
		case Int64:
			RowAdd(t.Int32Data, other.Int64Data, out.Int64Data, t.Shape)
		}
	} else if t.DType == Int64 && other.DType == Float {
		switch out.DType {
		case Float:
			RowAdd(t.Int64Data, other.FloatData, out.FloatData, t.Shape)
		case Double:
			RowAdd(t.Int64Data, other.FloatData, out.DoubleData, t.Shape)
		case Int32:
			RowAdd(t.Int64Data, other.FloatData, out.Int32Data, t.Shape)
		case Int64:
			RowAdd(t.Int64Data, other.FloatData, out.Int64Data, t.Shape)
		}
	} else if t.DType == Int64 && other.DType == Double {
		switch out.DType {
		case Float:
			RowAdd(t.Int64Data, other.DoubleData, out.FloatData, t.Shape)
		case Double:
			RowAdd(t.Int64Data, other.DoubleData, out.DoubleData, t.Shape)
		case Int32:
			RowAdd(t.Int64Data, other.DoubleData, out.Int32Data, t.Shape)
		case Int64:
			RowAdd(t.Int64Data, other.DoubleData, out.Int64Data, t.Shape)
		}
	} else if t.DType == Int64 && other.DType == Int32 {
		switch out.DType {
		case Float:
			RowAdd(t.Int64Data, other.Int32Data, out.FloatData, t.Shape)
		case Double:
			RowAdd(t.Int64Data, other.Int32Data, out.DoubleData, t.Shape)
		case Int32:
			RowAdd(t.Int64Data, other.Int32Data, out.Int32Data, t.Shape)
		case Int64:
			RowAdd(t.Int64Data, other.Int32Data, out.Int64Data, t.Shape)
		}
	} else if t.DType == Int64 && other.DType == Int64 {
		switch out.DType {
		case Float:
			RowAdd(t.Int64Data, other.Int64Data, out.FloatData, t.Shape)
		case Double:
			RowAdd(t.Int64Data, other.Int64Data, out.DoubleData, t.Shape)
		case Int32:
			RowAdd(t.Int64Data, other.Int64Data, out.Int32Data, t.Shape)
		case Int64:
			RowAdd(t.Int64Data, other.Int64Data, out.Int64Data, t.Shape)
		}
	}
	return out, nil
}

func (t *Tensor) addCol(other *Tensor, out *Tensor) (*Tensor, error) {
	colLength := other.Shape[0]
	if colLength != t.Shape[0] {
		return nil, fmt.Errorf("operants cols don't align: %d != %d", colLength, t.Shape[0])
	}
	if out == nil {
		out = createOutputTensor(t.DType, other.DType, t.Shape)
	}
	if t.DType == Float && other.DType == Float {
		switch out.DType {
		case Float:
			ColAdd(t.FloatData, other.FloatData, out.FloatData, t.Shape)
		case Double:
			ColAdd(t.FloatData, other.FloatData, out.DoubleData, t.Shape)
		case Int32:
			ColAdd(t.FloatData, other.FloatData, out.Int32Data, t.Shape)
		case Int64:
			ColAdd(t.FloatData, other.FloatData, out.Int64Data, t.Shape)
		}
	} else if t.DType == Float && other.DType == Double {
		switch out.DType {
		case Float:
			ColAdd(t.FloatData, other.DoubleData, out.FloatData, t.Shape)
		case Double:
			ColAdd(t.FloatData, other.DoubleData, out.DoubleData, t.Shape)
		case Int32:
			ColAdd(t.FloatData, other.DoubleData, out.Int32Data, t.Shape)
		case Int64:
			ColAdd(t.FloatData, other.DoubleData, out.Int64Data, t.Shape)
		}
	} else if t.DType == Float && other.DType == Int32 {
		switch out.DType {
		case Float:
			ColAdd(t.FloatData, other.Int32Data, out.FloatData, t.Shape)
		case Double:
			ColAdd(t.FloatData, other.Int32Data, out.DoubleData, t.Shape)
		case Int32:
			ColAdd(t.FloatData, other.Int32Data, out.Int32Data, t.Shape)
		case Int64:
			ColAdd(t.FloatData, other.Int32Data, out.Int64Data, t.Shape)
		}
	} else if t.DType == Float && other.DType == Int64 {
		switch out.DType {
		case Float:
			ColAdd(t.FloatData, other.Int64Data, out.FloatData, t.Shape)
		case Double:
			ColAdd(t.FloatData, other.Int64Data, out.DoubleData, t.Shape)
		case Int32:
			ColAdd(t.FloatData, other.Int64Data, out.Int32Data, t.Shape)
		case Int64:
			ColAdd(t.FloatData, other.Int64Data, out.Int64Data, t.Shape)
		}
	} else if t.DType == Double && other.DType == Float {
		switch out.DType {
		case Float:
			ColAdd(t.DoubleData, other.FloatData, out.FloatData, t.Shape)
		case Double:
			ColAdd(t.DoubleData, other.FloatData, out.DoubleData, t.Shape)
		case Int32:
			ColAdd(t.DoubleData, other.FloatData, out.Int32Data, t.Shape)
		case Int64:
			ColAdd(t.DoubleData, other.FloatData, out.Int64Data, t.Shape)
		}
	} else if t.DType == Double && other.DType == Double {
		switch out.DType {
		case Float:
			ColAdd(t.DoubleData, other.DoubleData, out.FloatData, t.Shape)
		case Double:
			ColAdd(t.DoubleData, other.DoubleData, out.DoubleData, t.Shape)
		case Int32:
			ColAdd(t.DoubleData, other.DoubleData, out.Int32Data, t.Shape)
		case Int64:
			ColAdd(t.DoubleData, other.DoubleData, out.Int64Data, t.Shape)
		}
	} else if t.DType == Double && other.DType == Int32 {
		switch out.DType {
		case Float:
			ColAdd(t.DoubleData, other.Int32Data, out.FloatData, t.Shape)
		case Double:
			ColAdd(t.DoubleData, other.Int32Data, out.DoubleData, t.Shape)
		case Int32:
			ColAdd(t.DoubleData, other.Int32Data, out.Int32Data, t.Shape)
		case Int64:
			ColAdd(t.DoubleData, other.Int32Data, out.Int64Data, t.Shape)
		}
	} else if t.DType == Double && other.DType == Int64 {
		switch out.DType {
		case Float:
			ColAdd(t.DoubleData, other.Int64Data, out.FloatData, t.Shape)
		case Double:
			ColAdd(t.DoubleData, other.Int64Data, out.DoubleData, t.Shape)
		case Int32:
			ColAdd(t.DoubleData, other.Int64Data, out.Int32Data, t.Shape)
		case Int64:
			ColAdd(t.DoubleData, other.Int64Data, out.Int64Data, t.Shape)
		}
	} else if t.DType == Int32 && other.DType == Float {
		switch out.DType {
		case Float:
			ColAdd(t.Int32Data, other.FloatData, out.FloatData, t.Shape)
		case Double:
			ColAdd(t.Int32Data, other.FloatData, out.DoubleData, t.Shape)
		case Int32:
			ColAdd(t.Int32Data, other.FloatData, out.Int32Data, t.Shape)
		case Int64:
			ColAdd(t.Int32Data, other.FloatData, out.Int64Data, t.Shape)
		}
	} else if t.DType == Int32 && other.DType == Double {
		switch out.DType {
		case Float:
			ColAdd(t.Int32Data, other.DoubleData, out.FloatData, t.Shape)
		case Double:
			ColAdd(t.Int32Data, other.DoubleData, out.DoubleData, t.Shape)
		case Int32:
			ColAdd(t.Int32Data, other.DoubleData, out.Int32Data, t.Shape)
		case Int64:
			ColAdd(t.Int32Data, other.DoubleData, out.Int64Data, t.Shape)
		}
	} else if t.DType == Int32 && other.DType == Int32 {
		switch out.DType {
		case Float:
			ColAdd(t.Int32Data, other.Int32Data, out.FloatData, t.Shape)
		case Double:
			ColAdd(t.Int32Data, other.Int32Data, out.DoubleData, t.Shape)
		case Int32:
			ColAdd(t.Int32Data, other.Int32Data, out.Int32Data, t.Shape)
		case Int64:
			ColAdd(t.Int32Data, other.Int32Data, out.Int64Data, t.Shape)
		}
	} else if t.DType == Int32 && other.DType == Int64 {
		switch out.DType {
		case Float:
			ColAdd(t.Int32Data, other.Int64Data, out.FloatData, t.Shape)
		case Double:
			ColAdd(t.Int32Data, other.Int64Data, out.DoubleData, t.Shape)
		case Int32:
			ColAdd(t.Int32Data, other.Int64Data, out.Int32Data, t.Shape)
		case Int64:
			ColAdd(t.Int32Data, other.Int64Data, out.Int64Data, t.Shape)
		}
	} else if t.DType == Int64 && other.DType == Float {
		switch out.DType {
		case Float:
			ColAdd(t.Int64Data, other.FloatData, out.FloatData, t.Shape)
		case Double:
			ColAdd(t.Int64Data, other.FloatData, out.DoubleData, t.Shape)
		case Int32:
			ColAdd(t.Int64Data, other.FloatData, out.Int32Data, t.Shape)
		case Int64:
			ColAdd(t.Int64Data, other.FloatData, out.Int64Data, t.Shape)
		}
	} else if t.DType == Int64 && other.DType == Double {
		switch out.DType {
		case Float:
			ColAdd(t.Int64Data, other.DoubleData, out.FloatData, t.Shape)
		case Double:
			ColAdd(t.Int64Data, other.DoubleData, out.DoubleData, t.Shape)
		case Int32:
			ColAdd(t.Int64Data, other.DoubleData, out.Int32Data, t.Shape)
		case Int64:
			ColAdd(t.Int64Data, other.DoubleData, out.Int64Data, t.Shape)
		}
	} else if t.DType == Int64 && other.DType == Int32 {
		switch out.DType {
		case Float:
			ColAdd(t.Int64Data, other.Int32Data, out.FloatData, t.Shape)
		case Double:
			ColAdd(t.Int64Data, other.Int32Data, out.DoubleData, t.Shape)
		case Int32:
			ColAdd(t.Int64Data, other.Int32Data, out.Int32Data, t.Shape)
		case Int64:
			ColAdd(t.Int64Data, other.Int32Data, out.Int64Data, t.Shape)
		}
	} else if t.DType == Int64 && other.DType == Int64 {
		switch out.DType {
		case Float:
			ColAdd(t.Int64Data, other.Int64Data, out.FloatData, t.Shape)
		case Double:
			ColAdd(t.Int64Data, other.Int64Data, out.DoubleData, t.Shape)
		case Int32:
			ColAdd(t.Int64Data, other.Int64Data, out.Int32Data, t.Shape)
		case Int64:
			ColAdd(t.Int64Data, other.Int64Data, out.Int64Data, t.Shape)
		}
	}
	return out, nil
}
