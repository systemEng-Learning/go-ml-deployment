package tensor

import "errors"

func (t *Tensor) Add(other *Tensor, out *Tensor) (*Tensor, error) {
	if out == nil || len(t.Shape) != 2 || len(other.Shape) != 1 || t.DType != other.DType || t.DType != Double {
		return nil, errors.ErrUnsupported
	}

	if t.Shape[1] != other.Shape[0] {
		return nil, errors.ErrUnsupported
	}

	for i := range t.Shape[0] {
		for j := range t.Shape[1] {
			out.DoubleData[i*t.Shape[1]+j] = t.DoubleData[i*t.Shape[1]+j] + other.DoubleData[j]
		}
	}
	return out, nil
}
