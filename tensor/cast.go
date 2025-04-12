package tensor

import (
	"log"
)

func cast[From Numeric, To Numeric](from []From, to []To, length int) []To {
	if len(to) < length {
		to = make([]To, length)
	}
	for i := range length {
		to[i] = To(from[i])
	}
	return to
}

func (t *Tensor) Cast(to DataType) {
	if t.DType == IntMap || t.DType == StringMap {
		log.Println("casting map-like tensors isn't supported")
		return
	}
	if t.DType == to {
		return
	}
	length := t.Shape[0]
	if len(t.Shape) == 2 {
		length *= t.Shape[1]
	}

	if t.DType == Float && to == Double {
		t.DoubleData = cast(t.FloatData, t.DoubleData, length)
	} else if t.DType == Float && to == Int32 {
		t.Int32Data = cast(t.FloatData, t.Int32Data, length)
	} else if t.DType == Float && to == Int64 {
		t.Int64Data = cast(t.FloatData, t.Int64Data, length)
	} else if t.DType == Double && to == Float {
		t.FloatData = cast(t.DoubleData, t.FloatData, length)
	} else if t.DType == Double && to == Int32 {
		t.Int32Data = cast(t.DoubleData, t.Int32Data, length)
	} else if t.DType == Double && to == Int64 {
		t.Int64Data = cast(t.DoubleData, t.Int64Data, length)
	} else if t.DType == Int32 && to == Float {
		t.FloatData = cast(t.Int32Data, t.FloatData, length)
	} else if t.DType == Int32 && to == Double {
		t.DoubleData = cast(t.Int32Data, t.DoubleData, length)
	} else if t.DType == Int32 && to == Int64 {
		t.Int64Data = cast(t.Int32Data, t.Int64Data, length)
	} else if t.DType == Int64 && to == Float {
		t.FloatData = cast(t.Int64Data, t.FloatData, length)
	} else if t.DType == Int64 && to == Double {
		t.DoubleData = cast(t.Int64Data, t.DoubleData, length)
	} else if t.DType == Int64 && to == Int32 {
		t.Int32Data = cast(t.Int64Data, t.Int32Data, length)
	} else {
		log.Fatalf("unsupported cast combination: %v -> %v", t.DType, to)
	}
	t.DType = to
}
