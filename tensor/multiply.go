package tensors

import (
	"errors"
	"fmt"
)

func (t *Tensor) Dot(other *Tensor, out *Tensor) (*Tensor, error) {
	if out == nil || len(t.Shape) != len(other.Shape) || len(t.Shape) == 1 || t.DType != other.DType || t.DType != Double {
		return nil, errors.ErrUnsupported
	}
	if t.Shape[1] != other.Shape[1] {
		return nil, fmt.Errorf("cannot multiply 2 matrices with differing shape %v and %v", t.Shape, other.Shape)
	}
	for a := range t.Shape[0] {
		for c := range other.Shape[0] {
			for b := range t.Shape[1] {
				out.DoubleData[a*other.Shape[0]+c] += t.DoubleData[a*t.Shape[1]+b] * other.DoubleData[c*other.Shape[1]+b]
			}
		}
	}
	return out, nil
}
