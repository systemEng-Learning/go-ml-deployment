package tensor

import (
	"errors"
	"log"
)

func (t *Tensor) Cast(to DataType) {
	if t.DType == IntMap || t.DType == StringMap {
		log.Println("casting map-like tensors isn't supported")
		return
	}
	if t.DType == to {
		return
	}
	if t.DType == Float && to == Double {
		t.castFloattoDouble()
	} else if t.DType == Double && to == Float {
		t.castDoubleToFloat()
	} else {
		log.Fatal(errors.ErrUnsupported)
	}
}

func (t *Tensor) castFloattoDouble() {
	if len(t.DoubleData) < len(t.FloatData) {
		t.DoubleData = make([]float64, len(t.FloatData))
	}
	for i := range t.FloatData {
		t.DoubleData[i] = float64(t.FloatData[i])
	}
	t.DType = Double
}

func (t *Tensor) castDoubleToFloat() {
	if len(t.FloatData) < len(t.DoubleData) {
		t.FloatData = make([]float32, len(t.DoubleData))
	}
	for i := range t.DoubleData {
		t.FloatData[i] = float32(t.DoubleData[i])
	}
	t.DType = Float
}
