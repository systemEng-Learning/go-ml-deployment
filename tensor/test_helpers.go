package tensor

func mustTensor(t *Tensor, data any) *Tensor {
	switch t.DType {
	case Float:
		t.FloatData = data.([]float32)
	case Double:
		t.DoubleData = data.([]float64)
	case Int32:
		t.Int32Data = data.([]int32)
	case Int64:
		t.Int64Data = data.([]int64)
	}
	return t
}

func extractData(t *Tensor) any {
	switch t.DType {
	case Float:
		return t.FloatData
	case Double:
		return t.DoubleData
	case Int32:
		return t.Int32Data
	case Int64:
		return t.Int64Data
	}
	return nil
}
