package tensor

import (
	"errors"
	"fmt"
)

/*
*
This function is going to replicate some of the behaviours of the numpy dot operation. Here's how it will work
If both t and other are 1-D tensors, the result will be an inner product of vectors
If both t and other are 2-D tensors, it is a matrix multiplication with a difference. Let me explain:
Standard matrix multiplication works with 2 matrix where the first has an [M, N] shape and the second has an [N, T] shape.
This will result in an [M, T] shape. This is good but not so cache friendly. Instead, our dot function requires that the
second matrix has a [T, N] shape i.e transposed. It will still result in a [M, T] matrix.
If t is 2-D and other is 1-D, it will be a sum product over the last axis of t and other.
*
*/
func (t *Tensor) Dot(other *Tensor, out *Tensor) (*Tensor, error) {
	// Cannot multiply maps
	if t.DType == IntMap || t.DType == StringMap || other.DType == IntMap || other.DType == StringMap {
		return nil, errors.New("cannot execute dot operations on map tensors")
	}
	// Vector multiplication
	if len(t.Shape) == 1 && len(other.Shape) == 1 {
		return t.vectorMultiply(other, out)
	}
	if len(t.Shape) == 2 && len(other.Shape) == 2 {
		return t.matrixMultiply(other, out)
	}
	if len(t.Shape) == 2 && len(other.Shape) == 1 {
		return t.matrixVectorMultiply(other, out)
	}
	return nil, errors.ErrUnsupported
}

func (t *Tensor) vectorMultiply(other *Tensor, out *Tensor) (*Tensor, error) {
	// Both input tensors have to be of the same length
	if t.Shape[0] != other.Shape[0] {
		return nil, errors.New("both tensors are required to have equal length")
	}
	if out == nil {
		out = createOutputTensor(t.DType, other.DType, []int{1})
	}
	if out.DType == Float || out.DType == Double {
		var sum float64
		if t.DType == Float && other.DType == Float {
			for i := range t.Shape[0] {
				sum += float64(t.FloatData[i]) * float64(other.FloatData[i])
			}
		} else if t.DType == Float && other.DType == Double {
			for i := range t.Shape[0] {
				sum += float64(t.FloatData[i]) * other.DoubleData[i]
			}
		} else if t.DType == Float && other.DType == Int32 {
			for i := range t.Shape[0] {
				sum += float64(t.FloatData[i]) * float64(other.Int32Data[i])
			}
		} else if t.DType == Float && other.DType == Int64 {
			for i := range t.Shape[0] {
				sum += float64(t.FloatData[i]) * float64(other.Int64Data[i])
			}
		} else if t.DType == Double && other.DType == Float {
			for i := range t.Shape[0] {
				sum += t.DoubleData[i] * float64(other.FloatData[i])
			}
		} else if t.DType == Double && other.DType == Double {
			for i := range t.Shape[0] {
				sum += t.DoubleData[i] * other.DoubleData[i]
			}
		} else if t.DType == Double && other.DType == Int32 {
			for i := range t.Shape[0] {
				sum += t.DoubleData[i] * float64(other.Int32Data[i])
			}
		} else if t.DType == Double && other.DType == Int64 {
			for i := range t.Shape[0] {
				sum += t.DoubleData[i] * float64(other.Int64Data[i])
			}
		} else if t.DType == Int32 && other.DType == Float {
			for i := range t.Shape[0] {
				sum += float64(t.Int32Data[i]) * float64(other.FloatData[i])
			}
		} else if t.DType == Int32 && other.DType == Double {
			for i := range t.Shape[0] {
				sum += float64(t.Int32Data[i]) * other.DoubleData[i]
			}
		} else if t.DType == Int32 && other.DType == Int32 {
			for i := range t.Shape[0] {
				sum += float64(t.Int32Data[i]) * float64(other.Int32Data[i])
			}
		} else if t.DType == Int32 && other.DType == Int64 {
			for i := range t.Shape[0] {
				sum += float64(t.Int32Data[i]) * float64(other.Int64Data[i])
			}
		} else if t.DType == Int64 && other.DType == Float {
			for i := range t.Shape[0] {
				sum += float64(t.Int64Data[i]) * float64(other.FloatData[i])
			}
		} else if t.DType == Int64 && other.DType == Double {
			for i := range t.Shape[0] {
				sum += float64(t.Int64Data[i]) * other.DoubleData[i]
			}
		} else if t.DType == Int64 && other.DType == Int32 {
			for i := range t.Shape[0] {
				sum += float64(t.Int64Data[i]) * float64(other.Int32Data[i])
			}
		} else if t.DType == Int64 && other.DType == Int64 {
			for i := range t.Shape[0] {
				sum += float64(t.Int64Data[i] * other.Int64Data[i])
			}
		}
		if out.DType == Float {
			out.FloatData[0] = float32(sum)
		} else {
			out.DoubleData[0] = sum
		}
	} else if out.DType == Int32 || out.DType == Int64 {
		var sum int64
		if t.DType == Float && other.DType == Float {
			for i := range t.Shape[0] {
				sum += int64(t.FloatData[i]) * int64(other.FloatData[i])
			}
		} else if t.DType == Float && other.DType == Double {
			for i := range t.Shape[0] {
				sum += int64(t.FloatData[i]) * int64(other.DoubleData[i])
			}
		} else if t.DType == Float && other.DType == Int32 {
			for i := range t.Shape[0] {
				sum += int64(t.FloatData[i]) * int64(other.Int32Data[i])
			}
		} else if t.DType == Float && other.DType == Int64 {
			for i := range t.Shape[0] {
				sum += int64(t.FloatData[i]) * other.Int64Data[i]
			}
		} else if t.DType == Double && other.DType == Float {
			for i := range t.Shape[0] {
				sum += int64(t.DoubleData[i]) * int64(other.FloatData[i])
			}
		} else if t.DType == Double && other.DType == Double {
			for i := range t.Shape[0] {
				sum += int64(t.DoubleData[i]) * int64(other.DoubleData[i])
			}
		} else if t.DType == Double && other.DType == Int32 {
			for i := range t.Shape[0] {
				sum += int64(t.DoubleData[i]) * int64(other.Int32Data[i])
			}
		} else if t.DType == Double && other.DType == Int64 {
			for i := range t.Shape[0] {
				sum += int64(t.DoubleData[i]) * other.Int64Data[i]
			}
		} else if t.DType == Int32 && other.DType == Float {
			for i := range t.Shape[0] {
				sum += int64(t.Int32Data[i]) * int64(other.FloatData[i])
			}
		} else if t.DType == Int32 && other.DType == Double {
			for i := range t.Shape[0] {
				sum += int64(t.Int32Data[i]) * int64(other.DoubleData[i])
			}
		} else if t.DType == Int32 && other.DType == Int32 {
			for i := range t.Shape[0] {
				sum += int64(t.Int32Data[i]) * int64(other.Int32Data[i])
			}
		} else if t.DType == Int32 && other.DType == Int64 {
			for i := range t.Shape[0] {
				sum += int64(t.Int32Data[i]) * other.Int64Data[i]
			}
		} else if t.DType == Int64 && other.DType == Float {
			for i := range t.Shape[0] {
				sum += t.Int64Data[i] * int64(other.FloatData[i])
			}
		} else if t.DType == Int64 && other.DType == Double {
			for i := range t.Shape[0] {
				sum += t.Int64Data[i] * int64(other.DoubleData[i])
			}
		} else if t.DType == Int64 && other.DType == Int32 {
			for i := range t.Shape[0] {
				sum += t.Int64Data[i] * int64(other.Int32Data[i])
			}
		} else if t.DType == Int64 && other.DType == Int64 {
			for i := range t.Shape[0] {
				sum += t.Int64Data[i] * other.Int64Data[i]
			}
		}
		if out.DType == Int32 {
			out.Int32Data[0] = int32(sum)
		} else {
			out.Int64Data[0] = sum
		}
	}
	return out, nil
}

func (t *Tensor) matrixMultiply(other *Tensor, out *Tensor) (*Tensor, error) {
	// Ensure both have the same number of columns
	if t.Shape[1] != other.Shape[1] {
		return nil, errors.New("both matrix column axis are not equal")
	}
	if out == nil {
		out = createOutputTensor(t.DType, other.DType, []int{t.Shape[0], other.Shape[0]})
	}

	if out.DType == Float {
		var sum float64
		if t.DType == Float && other.DType == Float {
			for a := range t.Shape[0] {
				for c := range other.Shape[0] {
					sum = 0
					for b := range t.Shape[1] {
						sum += float64(t.FloatData[a*t.Shape[1]+b]) * float64(other.FloatData[c*other.Shape[1]+b])
					}
					out.FloatData[a*other.Shape[0]+c] = float32(sum)
				}
			}
		} else if t.DType == Float && other.DType == Double {
			for a := range t.Shape[0] {
				for c := range other.Shape[0] {
					sum = 0
					for b := range t.Shape[1] {
						sum += float64(t.FloatData[a*t.Shape[1]+b]) * other.DoubleData[c*other.Shape[1]+b]
					}
					out.FloatData[a*other.Shape[0]+c] = float32(sum)
				}
			}
		} else if t.DType == Float && other.DType == Int32 {
			for a := range t.Shape[0] {
				for c := range other.Shape[0] {
					sum = 0
					for b := range t.Shape[1] {
						sum += float64(t.FloatData[a*t.Shape[1]+b]) * float64(other.Int32Data[c*other.Shape[1]+b])
					}
					out.FloatData[a*other.Shape[0]+c] = float32(sum)
				}
			}
		} else if t.DType == Float && other.DType == Int64 {
			for a := range t.Shape[0] {
				for c := range other.Shape[0] {
					sum = 0
					for b := range t.Shape[1] {
						sum += float64(t.FloatData[a*t.Shape[1]+b]) * float64(other.Int64Data[c*other.Shape[1]+b])
					}
					out.FloatData[a*other.Shape[0]+c] = float32(sum)
				}
			}
		} else if t.DType == Double && other.DType == Float {
			for a := range t.Shape[0] {
				for c := range other.Shape[0] {
					sum = 0
					for b := range t.Shape[1] {
						sum += t.DoubleData[a*t.Shape[1]+b] * float64(other.FloatData[c*other.Shape[1]+b])
					}
					out.FloatData[a*other.Shape[0]+c] = float32(sum)
				}
			}
		} else if t.DType == Double && other.DType == Double {
			for a := range t.Shape[0] {
				for c := range other.Shape[0] {
					sum = 0
					for b := range t.Shape[1] {
						sum += t.DoubleData[a*t.Shape[1]+b] * other.DoubleData[c*other.Shape[1]+b]
					}
					out.FloatData[a*other.Shape[0]+c] = float32(sum)
				}
			}
		} else if t.DType == Double && other.DType == Int32 {
			for a := range t.Shape[0] {
				for c := range other.Shape[0] {
					sum = 0
					for b := range t.Shape[1] {
						sum += t.DoubleData[a*t.Shape[1]+b] * float64(other.Int32Data[c*other.Shape[1]+b])
					}
					out.FloatData[a*other.Shape[0]+c] = float32(sum)
				}
			}
		} else if t.DType == Double && other.DType == Int64 {
			for a := range t.Shape[0] {
				for c := range other.Shape[0] {
					sum = 0
					for b := range t.Shape[1] {
						sum += t.DoubleData[a*t.Shape[1]+b] * float64(other.Int64Data[c*other.Shape[1]+b])
					}
					out.FloatData[a*other.Shape[0]+c] = float32(sum)
				}
			}
		} else if t.DType == Int32 && other.DType == Float {
			for a := range t.Shape[0] {
				for c := range other.Shape[0] {
					sum = 0
					for b := range t.Shape[1] {
						sum += float64(t.Int32Data[a*t.Shape[1]+b]) * float64(other.FloatData[c*other.Shape[1]+b])
					}
					out.FloatData[a*other.Shape[0]+c] = float32(sum)
				}
			}
		} else if t.DType == Int32 && other.DType == Double {
			for a := range t.Shape[0] {
				for c := range other.Shape[0] {
					sum = 0
					for b := range t.Shape[1] {
						sum += float64(t.Int32Data[a*t.Shape[1]+b]) * other.DoubleData[c*other.Shape[1]+b]
					}
					out.FloatData[a*other.Shape[0]+c] = float32(sum)
				}
			}
		} else if t.DType == Int32 && other.DType == Int32 {
			for a := range t.Shape[0] {
				for c := range other.Shape[0] {
					sum = 0
					for b := range t.Shape[1] {
						sum += float64(t.Int32Data[a*t.Shape[1]+b]) * float64(other.Int32Data[c*other.Shape[1]+b])
					}
					out.FloatData[a*other.Shape[0]+c] = float32(sum)
				}
			}
		} else if t.DType == Int32 && other.DType == Int64 {
			for a := range t.Shape[0] {
				for c := range other.Shape[0] {
					sum = 0
					for b := range t.Shape[1] {
						sum += float64(t.Int32Data[a*t.Shape[1]+b]) * float64(other.Int64Data[c*other.Shape[1]+b])
					}
					out.FloatData[a*other.Shape[0]+c] = float32(sum)
				}
			}
		} else if t.DType == Int64 && other.DType == Float {
			for a := range t.Shape[0] {
				for c := range other.Shape[0] {
					sum = 0
					for b := range t.Shape[1] {
						sum += float64(t.Int64Data[a*t.Shape[1]+b]) * float64(other.FloatData[c*other.Shape[1]+b])
					}
					out.FloatData[a*other.Shape[0]+c] = float32(sum)
				}
			}
		} else if t.DType == Int64 && other.DType == Double {
			for a := range t.Shape[0] {
				for c := range other.Shape[0] {
					sum = 0
					for b := range t.Shape[1] {
						sum += float64(t.Int64Data[a*t.Shape[1]+b]) * other.DoubleData[c*other.Shape[1]+b]
					}
					out.FloatData[a*other.Shape[0]+c] = float32(sum)
				}
			}
		} else if t.DType == Int64 && other.DType == Int32 {
			for a := range t.Shape[0] {
				for c := range other.Shape[0] {
					sum = 0
					for b := range t.Shape[1] {
						sum += float64(t.Int64Data[a*t.Shape[1]+b]) * float64(other.Int32Data[c*other.Shape[1]+b])
					}
					out.FloatData[a*other.Shape[0]+c] = float32(sum)
				}
			}
		} else if t.DType == Int64 && other.DType == Int64 {
			for a := range t.Shape[0] {
				for c := range other.Shape[0] {
					sum = 0
					for b := range t.Shape[1] {
						sum += float64(t.Int64Data[a*t.Shape[1]+b]) * float64(other.Int64Data[c*other.Shape[1]+b])
					}
					out.FloatData[a*other.Shape[0]+c] = float32(sum)
				}
			}
		}
	} else if out.DType == Double {
		var sum float64
		if t.DType == Float && other.DType == Float {
			for a := range t.Shape[0] {
				for c := range other.Shape[0] {
					sum = 0
					for b := range t.Shape[1] {
						sum += float64(t.FloatData[a*t.Shape[1]+b]) * float64(other.FloatData[c*other.Shape[1]+b])
					}
					out.DoubleData[a*other.Shape[0]+c] = sum
				}
			}
		} else if t.DType == Float && other.DType == Double {
			for a := range t.Shape[0] {
				for c := range other.Shape[0] {
					sum = 0
					for b := range t.Shape[1] {
						sum += float64(t.FloatData[a*t.Shape[1]+b]) * other.DoubleData[c*other.Shape[1]+b]
					}
					out.DoubleData[a*other.Shape[0]+c] = sum
				}
			}
		} else if t.DType == Float && other.DType == Int32 {
			for a := range t.Shape[0] {
				for c := range other.Shape[0] {
					sum = 0
					for b := range t.Shape[1] {
						sum += float64(t.FloatData[a*t.Shape[1]+b]) * float64(other.Int32Data[c*other.Shape[1]+b])
					}
					out.DoubleData[a*other.Shape[0]+c] = sum
				}
			}
		} else if t.DType == Float && other.DType == Int64 {
			for a := range t.Shape[0] {
				for c := range other.Shape[0] {
					sum = 0
					for b := range t.Shape[1] {
						sum += float64(t.FloatData[a*t.Shape[1]+b]) * float64(other.Int64Data[c*other.Shape[1]+b])
					}
					out.DoubleData[a*other.Shape[0]+c] = sum
				}
			}
		} else if t.DType == Double && other.DType == Float {
			for a := range t.Shape[0] {
				for c := range other.Shape[0] {
					sum = 0
					for b := range t.Shape[1] {
						sum += t.DoubleData[a*t.Shape[1]+b] * float64(other.FloatData[c*other.Shape[1]+b])
					}
					out.DoubleData[a*other.Shape[0]+c] = sum
				}
			}
		} else if t.DType == Double && other.DType == Double {
			for a := range t.Shape[0] {
				for c := range other.Shape[0] {
					sum = 0
					for b := range t.Shape[1] {
						sum += t.DoubleData[a*t.Shape[1]+b] * other.DoubleData[c*other.Shape[1]+b]
					}
					out.DoubleData[a*other.Shape[0]+c] = sum
				}
			}
		} else if t.DType == Double && other.DType == Int32 {
			for a := range t.Shape[0] {
				for c := range other.Shape[0] {
					sum = 0
					for b := range t.Shape[1] {
						sum += t.DoubleData[a*t.Shape[1]+b] * float64(other.Int32Data[c*other.Shape[1]+b])
					}
					out.DoubleData[a*other.Shape[0]+c] = sum
				}
			}
		} else if t.DType == Double && other.DType == Int64 {
			for a := range t.Shape[0] {
				for c := range other.Shape[0] {
					sum = 0
					for b := range t.Shape[1] {
						sum += t.DoubleData[a*t.Shape[1]+b] * float64(other.Int64Data[c*other.Shape[1]+b])
					}
					out.DoubleData[a*other.Shape[0]+c] = sum
				}
			}
		} else if t.DType == Int32 && other.DType == Float {
			for a := range t.Shape[0] {
				for c := range other.Shape[0] {
					sum = 0
					for b := range t.Shape[1] {
						sum += float64(t.Int32Data[a*t.Shape[1]+b]) * float64(other.FloatData[c*other.Shape[1]+b])
					}
					out.DoubleData[a*other.Shape[0]+c] = sum
				}
			}
		} else if t.DType == Int32 && other.DType == Double {
			for a := range t.Shape[0] {
				for c := range other.Shape[0] {
					sum = 0
					for b := range t.Shape[1] {
						sum += float64(t.Int32Data[a*t.Shape[1]+b]) * other.DoubleData[c*other.Shape[1]+b]
					}
					out.DoubleData[a*other.Shape[0]+c] = sum
				}
			}
		} else if t.DType == Int32 && other.DType == Int32 {
			for a := range t.Shape[0] {
				for c := range other.Shape[0] {
					sum = 0
					for b := range t.Shape[1] {
						sum += float64(t.Int32Data[a*t.Shape[1]+b]) * float64(other.Int32Data[c*other.Shape[1]+b])
					}
					out.DoubleData[a*other.Shape[0]+c] = sum
				}
			}
		} else if t.DType == Int32 && other.DType == Int64 {
			for a := range t.Shape[0] {
				for c := range other.Shape[0] {
					sum = 0
					for b := range t.Shape[1] {
						sum += float64(t.Int32Data[a*t.Shape[1]+b]) * float64(other.Int64Data[c*other.Shape[1]+b])
					}
					out.DoubleData[a*other.Shape[0]+c] = sum
				}
			}
		} else if t.DType == Int64 && other.DType == Float {
			for a := range t.Shape[0] {
				for c := range other.Shape[0] {
					sum = 0
					for b := range t.Shape[1] {
						sum += float64(t.Int64Data[a*t.Shape[1]+b]) * float64(other.FloatData[c*other.Shape[1]+b])
					}
					out.DoubleData[a*other.Shape[0]+c] = sum
				}
			}
		} else if t.DType == Int64 && other.DType == Double {
			for a := range t.Shape[0] {
				for c := range other.Shape[0] {
					sum = 0
					for b := range t.Shape[1] {
						sum += float64(t.Int64Data[a*t.Shape[1]+b]) * other.DoubleData[c*other.Shape[1]+b]
					}
					out.DoubleData[a*other.Shape[0]+c] = sum
				}
			}
		} else if t.DType == Int64 && other.DType == Int32 {
			for a := range t.Shape[0] {
				for c := range other.Shape[0] {
					sum = 0
					for b := range t.Shape[1] {
						sum += float64(t.Int64Data[a*t.Shape[1]+b]) * float64(other.Int32Data[c*other.Shape[1]+b])
					}
					out.DoubleData[a*other.Shape[0]+c] = sum
				}
			}
		} else if t.DType == Int64 && other.DType == Int64 {
			for a := range t.Shape[0] {
				for c := range other.Shape[0] {
					sum = 0
					for b := range t.Shape[1] {
						sum += float64(t.Int64Data[a*t.Shape[1]+b]) * float64(other.Int64Data[c*other.Shape[1]+b])
					}
					out.DoubleData[a*other.Shape[0]+c] = sum
				}
			}
		}
	} else if out.DType == Int32 {
		var sum int64
		if t.DType == Float && other.DType == Float {
			for a := range t.Shape[0] {
				for c := range other.Shape[0] {
					sum = 0
					for b := range t.Shape[1] {
						sum += int64(t.FloatData[a*t.Shape[1]+b]) * int64(other.FloatData[c*other.Shape[1]+b])
					}
					out.Int32Data[a*other.Shape[0]+c] = int32(sum)
				}
			}
		} else if t.DType == Float && other.DType == Double {
			for a := range t.Shape[0] {
				for c := range other.Shape[0] {
					sum = 0
					for b := range t.Shape[1] {
						sum += int64(t.FloatData[a*t.Shape[1]+b]) * int64(other.DoubleData[c*other.Shape[1]+b])
					}
					out.Int32Data[a*other.Shape[0]+c] = int32(sum)
				}
			}
		} else if t.DType == Float && other.DType == Int32 {
			for a := range t.Shape[0] {
				for c := range other.Shape[0] {
					sum = 0
					for b := range t.Shape[1] {
						sum += int64(t.FloatData[a*t.Shape[1]+b]) * int64(other.Int32Data[c*other.Shape[1]+b])
					}
					out.Int32Data[a*other.Shape[0]+c] = int32(sum)
				}
			}
		} else if t.DType == Float && other.DType == Int64 {
			for a := range t.Shape[0] {
				for c := range other.Shape[0] {
					sum = 0
					for b := range t.Shape[1] {
						sum += int64(t.FloatData[a*t.Shape[1]+b]) * other.Int64Data[c*other.Shape[1]+b]
					}
					out.Int32Data[a*other.Shape[0]+c] = int32(sum)
				}
			}
		} else if t.DType == Double && other.DType == Float {
			for a := range t.Shape[0] {
				for c := range other.Shape[0] {
					sum = 0
					for b := range t.Shape[1] {
						sum += int64(t.DoubleData[a*t.Shape[1]+b]) * int64(other.FloatData[c*other.Shape[1]+b])
					}
					out.Int32Data[a*other.Shape[0]+c] = int32(sum)
				}
			}
		} else if t.DType == Double && other.DType == Double {
			for a := range t.Shape[0] {
				for c := range other.Shape[0] {
					sum = 0
					for b := range t.Shape[1] {
						sum += int64(t.DoubleData[a*t.Shape[1]+b]) * int64(other.DoubleData[c*other.Shape[1]+b])
					}
					out.Int32Data[a*other.Shape[0]+c] = int32(sum)
				}
			}
		} else if t.DType == Double && other.DType == Int32 {
			for a := range t.Shape[0] {
				for c := range other.Shape[0] {
					sum = 0
					for b := range t.Shape[1] {
						sum += int64(t.DoubleData[a*t.Shape[1]+b]) * int64(other.Int32Data[c*other.Shape[1]+b])
					}
					out.Int32Data[a*other.Shape[0]+c] = int32(sum)
				}
			}
		} else if t.DType == Double && other.DType == Int64 {
			for a := range t.Shape[0] {
				for c := range other.Shape[0] {
					sum = 0
					for b := range t.Shape[1] {
						sum += int64(t.DoubleData[a*t.Shape[1]+b]) * other.Int64Data[c*other.Shape[1]+b]
					}
					out.Int32Data[a*other.Shape[0]+c] = int32(sum)
				}
			}
		} else if t.DType == Int32 && other.DType == Float {
			for a := range t.Shape[0] {
				for c := range other.Shape[0] {
					sum = 0
					for b := range t.Shape[1] {
						sum += int64(t.Int32Data[a*t.Shape[1]+b]) * int64(other.FloatData[c*other.Shape[1]+b])
					}
					out.Int32Data[a*other.Shape[0]+c] = int32(sum)
				}
			}
		} else if t.DType == Int32 && other.DType == Double {
			for a := range t.Shape[0] {
				for c := range other.Shape[0] {
					sum = 0
					for b := range t.Shape[1] {
						sum += int64(t.Int32Data[a*t.Shape[1]+b]) * int64(other.DoubleData[c*other.Shape[1]+b])
					}
					out.Int32Data[a*other.Shape[0]+c] = int32(sum)
				}
			}
		} else if t.DType == Int32 && other.DType == Int32 {
			for a := range t.Shape[0] {
				for c := range other.Shape[0] {
					sum = 0
					for b := range t.Shape[1] {
						sum += int64(t.Int32Data[a*t.Shape[1]+b]) * int64(other.Int32Data[c*other.Shape[1]+b])
					}
					out.Int32Data[a*other.Shape[0]+c] = int32(sum)
				}
			}
		} else if t.DType == Int32 && other.DType == Int64 {
			for a := range t.Shape[0] {
				for c := range other.Shape[0] {
					sum = 0
					for b := range t.Shape[1] {
						sum += int64(t.Int32Data[a*t.Shape[1]+b]) * other.Int64Data[c*other.Shape[1]+b]
					}
					out.Int32Data[a*other.Shape[0]+c] = int32(sum)
				}
			}
		} else if t.DType == Int64 && other.DType == Float {
			for a := range t.Shape[0] {
				for c := range other.Shape[0] {
					sum = 0
					for b := range t.Shape[1] {
						sum += t.Int64Data[a*t.Shape[1]+b] * int64(other.FloatData[c*other.Shape[1]+b])
					}
					out.Int32Data[a*other.Shape[0]+c] = int32(sum)
				}
			}
		} else if t.DType == Int64 && other.DType == Double {
			for a := range t.Shape[0] {
				for c := range other.Shape[0] {
					sum = 0
					for b := range t.Shape[1] {
						sum += t.Int64Data[a*t.Shape[1]+b] * int64(other.DoubleData[c*other.Shape[1]+b])
					}
					out.Int32Data[a*other.Shape[0]+c] = int32(sum)
				}
			}
		} else if t.DType == Int64 && other.DType == Int32 {
			for a := range t.Shape[0] {
				for c := range other.Shape[0] {
					sum = 0
					for b := range t.Shape[1] {
						sum += t.Int64Data[a*t.Shape[1]+b] * int64(other.Int32Data[c*other.Shape[1]+b])
					}
					out.Int32Data[a*other.Shape[0]+c] = int32(sum)
				}
			}
		} else if t.DType == Int64 && other.DType == Int64 {
			for a := range t.Shape[0] {
				for c := range other.Shape[0] {
					sum = 0
					for b := range t.Shape[1] {
						sum += t.Int64Data[a*t.Shape[1]+b] * other.Int64Data[c*other.Shape[1]+b]
					}
					out.Int32Data[a*other.Shape[0]+c] = int32(sum)
				}
			}
		}
	} else if out.DType == Int64 {
		var sum int64
		if t.DType == Float && other.DType == Float {
			for a := range t.Shape[0] {
				for c := range other.Shape[0] {
					sum = 0
					for b := range t.Shape[1] {
						sum += int64(t.FloatData[a*t.Shape[1]+b]) * int64(other.FloatData[c*other.Shape[1]+b])
					}
					out.Int64Data[a*other.Shape[0]+c] = sum
				}
			}
		} else if t.DType == Float && other.DType == Double {
			for a := range t.Shape[0] {
				for c := range other.Shape[0] {
					sum = 0
					for b := range t.Shape[1] {
						sum += int64(t.FloatData[a*t.Shape[1]+b]) * int64(other.DoubleData[c*other.Shape[1]+b])
					}
					out.Int64Data[a*other.Shape[0]+c] = sum
				}
			}
		} else if t.DType == Float && other.DType == Int32 {
			for a := range t.Shape[0] {
				for c := range other.Shape[0] {
					sum = 0
					for b := range t.Shape[1] {
						sum += int64(t.FloatData[a*t.Shape[1]+b]) * int64(other.Int32Data[c*other.Shape[1]+b])
					}
					out.Int64Data[a*other.Shape[0]+c] = sum
				}
			}
		} else if t.DType == Float && other.DType == Int64 {
			for a := range t.Shape[0] {
				for c := range other.Shape[0] {
					sum = 0
					for b := range t.Shape[1] {
						sum += int64(t.FloatData[a*t.Shape[1]+b]) * other.Int64Data[c*other.Shape[1]+b]
					}
					out.Int64Data[a*other.Shape[0]+c] = sum
				}
			}
		} else if t.DType == Double && other.DType == Float {
			for a := range t.Shape[0] {
				for c := range other.Shape[0] {
					sum = 0
					for b := range t.Shape[1] {
						sum += int64(t.DoubleData[a*t.Shape[1]+b]) * int64(other.FloatData[c*other.Shape[1]+b])
					}
					out.Int64Data[a*other.Shape[0]+c] = sum
				}
			}
		} else if t.DType == Double && other.DType == Double {
			for a := range t.Shape[0] {
				for c := range other.Shape[0] {
					sum = 0
					for b := range t.Shape[1] {
						sum += int64(t.DoubleData[a*t.Shape[1]+b]) * int64(other.DoubleData[c*other.Shape[1]+b])
					}
					out.Int64Data[a*other.Shape[0]+c] = sum
				}
			}
		} else if t.DType == Double && other.DType == Int32 {
			for a := range t.Shape[0] {
				for c := range other.Shape[0] {
					sum = 0
					for b := range t.Shape[1] {
						sum += int64(t.DoubleData[a*t.Shape[1]+b]) * int64(other.Int32Data[c*other.Shape[1]+b])
					}
					out.Int64Data[a*other.Shape[0]+c] = sum
				}
			}
		} else if t.DType == Double && other.DType == Int64 {
			for a := range t.Shape[0] {
				for c := range other.Shape[0] {
					sum = 0
					for b := range t.Shape[1] {
						sum += int64(t.DoubleData[a*t.Shape[1]+b]) * other.Int64Data[c*other.Shape[1]+b]
					}
					out.Int64Data[a*other.Shape[0]+c] = sum
				}
			}
		} else if t.DType == Int32 && other.DType == Float {
			for a := range t.Shape[0] {
				for c := range other.Shape[0] {
					sum = 0
					for b := range t.Shape[1] {
						sum += int64(t.Int32Data[a*t.Shape[1]+b]) * int64(other.FloatData[c*other.Shape[1]+b])
					}
					out.Int64Data[a*other.Shape[0]+c] = sum
				}
			}
		} else if t.DType == Int32 && other.DType == Double {
			for a := range t.Shape[0] {
				for c := range other.Shape[0] {
					sum = 0
					for b := range t.Shape[1] {
						sum += int64(t.Int32Data[a*t.Shape[1]+b]) * int64(other.DoubleData[c*other.Shape[1]+b])
					}
					out.Int64Data[a*other.Shape[0]+c] = sum
				}
			}
		} else if t.DType == Int32 && other.DType == Int32 {
			for a := range t.Shape[0] {
				for c := range other.Shape[0] {
					sum = 0
					for b := range t.Shape[1] {
						sum += int64(t.Int32Data[a*t.Shape[1]+b]) * int64(other.Int32Data[c*other.Shape[1]+b])
					}
					out.Int64Data[a*other.Shape[0]+c] = sum
				}
			}
		} else if t.DType == Int32 && other.DType == Int64 {
			for a := range t.Shape[0] {
				for c := range other.Shape[0] {
					sum = 0
					for b := range t.Shape[1] {
						sum += int64(t.Int32Data[a*t.Shape[1]+b]) * other.Int64Data[c*other.Shape[1]+b]
					}
					out.Int64Data[a*other.Shape[0]+c] = sum
				}
			}
		} else if t.DType == Int64 && other.DType == Float {
			for a := range t.Shape[0] {
				for c := range other.Shape[0] {
					sum = 0
					for b := range t.Shape[1] {
						sum += t.Int64Data[a*t.Shape[1]+b] * int64(other.FloatData[c*other.Shape[1]+b])
					}
					out.Int64Data[a*other.Shape[0]+c] = sum
				}
			}
		} else if t.DType == Int64 && other.DType == Double {
			for a := range t.Shape[0] {
				for c := range other.Shape[0] {
					sum = 0
					for b := range t.Shape[1] {
						sum += t.Int64Data[a*t.Shape[1]+b] * int64(other.DoubleData[c*other.Shape[1]+b])
					}
					out.Int64Data[a*other.Shape[0]+c] = sum
				}
			}
		} else if t.DType == Int64 && other.DType == Int32 {
			for a := range t.Shape[0] {
				for c := range other.Shape[0] {
					sum = 0
					for b := range t.Shape[1] {
						sum += t.Int64Data[a*t.Shape[1]+b] * int64(other.Int32Data[c*other.Shape[1]+b])
					}
					out.Int64Data[a*other.Shape[0]+c] = sum
				}
			}
		} else if t.DType == Int64 && other.DType == Int64 {
			for a := range t.Shape[0] {
				for c := range other.Shape[0] {
					sum = 0
					for b := range t.Shape[1] {
						sum += t.Int64Data[a*t.Shape[1]+b] * other.Int64Data[c*other.Shape[1]+b]
					}
					out.Int64Data[a*other.Shape[0]+c] = sum
				}
			}
		}
	}
	return out, nil
}

func (t *Tensor) matrixVectorMultiply(other *Tensor, out *Tensor) (*Tensor, error) {
	// Ensure the first matrix columns size is the same as the vector length
	if t.Shape[1] != other.Shape[0] {
		return nil, fmt.Errorf("unmatched tensor shape: %v, %v, %d != %d", t.Shape, other.Shape, t.Shape[1], other.Shape[0])
	}
	if out == nil {
		out = createOutputTensor(t.DType, other.DType, []int{t.Shape[0]})
	}
	if out.DType == Float || out.DType == Double {
		var sum float64
		if t.DType == Float && other.DType == Float {
			for i := range t.Shape[0] {
				sum = 0
				for j := range t.Shape[1] {
					sum += float64(t.FloatData[i*t.Shape[1]+j]) * float64(other.FloatData[j])
				}
				if out.DType == Float {
					out.FloatData[i] = float32(sum)
				} else {
					out.DoubleData[i] = sum
				}
			}
		} else if t.DType == Float && other.DType == Double {
			for i := range t.Shape[0] {
				sum = 0
				for j := range t.Shape[1] {
					sum += float64(t.FloatData[i*t.Shape[1]+j]) * other.DoubleData[j]
				}
				if out.DType == Float {
					out.FloatData[i] = float32(sum)
				} else {
					out.DoubleData[i] = sum
				}
			}
		} else if t.DType == Float && other.DType == Int32 {
			for i := range t.Shape[0] {
				sum = 0
				for j := range t.Shape[1] {
					sum += float64(t.FloatData[i*t.Shape[1]+j]) * float64(other.Int32Data[j])
				}
				if out.DType == Float {
					out.FloatData[i] = float32(sum)
				} else {
					out.DoubleData[i] = sum
				}
			}
		} else if t.DType == Float && other.DType == Int64 {
			for i := range t.Shape[0] {
				sum = 0
				for j := range t.Shape[1] {
					sum += float64(t.FloatData[i*t.Shape[1]+j]) * float64(other.Int64Data[j])
				}
				if out.DType == Float {
					out.FloatData[i] = float32(sum)
				} else {
					out.DoubleData[i] = sum
				}
			}
		} else if t.DType == Double && other.DType == Float {
			for i := range t.Shape[0] {
				sum = 0
				for j := range t.Shape[1] {
					sum += t.DoubleData[i*t.Shape[1]+j] * float64(other.FloatData[j])
				}
				if out.DType == Float {
					out.FloatData[i] = float32(sum)
				} else {
					out.DoubleData[i] = sum
				}
			}
		} else if t.DType == Double && other.DType == Double {
			for i := range t.Shape[0] {
				sum = 0
				for j := range t.Shape[1] {
					sum += t.DoubleData[i*t.Shape[1]+j] * other.DoubleData[j]
				}
				if out.DType == Float {
					out.FloatData[i] = float32(sum)
				} else {
					out.DoubleData[i] = sum
				}
			}
		} else if t.DType == Double && other.DType == Int32 {
			for i := range t.Shape[0] {
				sum = 0
				for j := range t.Shape[1] {
					sum += t.DoubleData[i*t.Shape[1]+j] * float64(other.Int32Data[j])
				}
				if out.DType == Float {
					out.FloatData[i] = float32(sum)
				} else {
					out.DoubleData[i] = sum
				}
			}
		} else if t.DType == Double && other.DType == Int64 {
			for i := range t.Shape[0] {
				sum = 0
				for j := range t.Shape[1] {
					sum += t.DoubleData[i*t.Shape[1]+j] * float64(other.Int64Data[j])
				}
				if out.DType == Float {
					out.FloatData[i] = float32(sum)
				} else {
					out.DoubleData[i] = sum
				}
			}
		} else if t.DType == Int32 && other.DType == Float {
			for i := range t.Shape[0] {
				sum = 0
				for j := range t.Shape[1] {
					sum += float64(t.Int32Data[i*t.Shape[1]+j]) * float64(other.FloatData[j])
				}
				if out.DType == Float {
					out.FloatData[i] = float32(sum)
				} else {
					out.DoubleData[i] = sum
				}
			}
		} else if t.DType == Int32 && other.DType == Double {
			for i := range t.Shape[0] {
				sum = 0
				for j := range t.Shape[1] {
					sum += float64(t.Int32Data[i*t.Shape[1]+j]) * other.DoubleData[j]
				}
				if out.DType == Float {
					out.FloatData[i] = float32(sum)
				} else {
					out.DoubleData[i] = sum
				}
			}
		} else if t.DType == Int32 && other.DType == Int32 {
			for i := range t.Shape[0] {
				sum = 0
				for j := range t.Shape[1] {
					sum += float64(t.Int32Data[i*t.Shape[1]+j]) * float64(other.Int32Data[j])
				}
				if out.DType == Float {
					out.FloatData[i] = float32(sum)
				} else {
					out.DoubleData[i] = sum
				}
			}
		} else if t.DType == Int32 && other.DType == Int64 {
			for i := range t.Shape[0] {
				sum = 0
				for j := range t.Shape[1] {
					sum += float64(t.Int32Data[i*t.Shape[1]+j]) * float64(other.Int64Data[j])
				}
				if out.DType == Float {
					out.FloatData[i] = float32(sum)
				} else {
					out.DoubleData[i] = sum
				}
			}
		} else if t.DType == Int64 && other.DType == Float {
			for i := range t.Shape[0] {
				sum = 0
				for j := range t.Shape[1] {
					sum += float64(t.Int64Data[i*t.Shape[1]+j]) * float64(other.FloatData[j])
				}
				if out.DType == Float {
					out.FloatData[i] = float32(sum)
				} else {
					out.DoubleData[i] = sum
				}
			}
		} else if t.DType == Int64 && other.DType == Double {
			for i := range t.Shape[0] {
				sum = 0
				for j := range t.Shape[1] {
					sum += float64(t.Int64Data[i*t.Shape[1]+j]) * other.DoubleData[j]
				}
				if out.DType == Float {
					out.FloatData[i] = float32(sum)
				} else {
					out.DoubleData[i] = sum
				}
			}
		} else if t.DType == Int64 && other.DType == Int32 {
			for i := range t.Shape[0] {
				sum = 0
				for j := range t.Shape[1] {
					sum += float64(t.Int64Data[i*t.Shape[1]+j]) * float64(other.Int32Data[j])
				}
				if out.DType == Float {
					out.FloatData[i] = float32(sum)
				} else {
					out.DoubleData[i] = sum
				}
			}
		} else if t.DType == Int64 && other.DType == Int64 {
			for i := range t.Shape[0] {
				sum = 0
				for j := range t.Shape[1] {
					sum += float64(t.Int64Data[i*t.Shape[1]+j]) * float64(other.Int64Data[j])
				}
				if out.DType == Float {
					out.FloatData[i] = float32(sum)
				} else {
					out.DoubleData[i] = sum
				}
			}
		}
	} else if out.DType == Int32 || out.DType == Int64 {
		var sum int64
		if t.DType == Float && other.DType == Float {
			for i := range t.Shape[0] {
				sum = 0
				for j := range t.Shape[1] {
					sum += int64(t.FloatData[i*t.Shape[1]+j]) * int64(other.FloatData[j])
				}
				if out.DType == Int32 {
					out.Int32Data[i] = int32(sum)
				} else {
					out.Int64Data[i] = sum
				}
			}
		} else if t.DType == Float && other.DType == Double {
			for i := range t.Shape[0] {
				sum = 0
				for j := range t.Shape[1] {
					sum += int64(t.FloatData[i*t.Shape[1]+j]) * int64(other.DoubleData[j])
				}
				if out.DType == Int32 {
					out.Int32Data[i] = int32(sum)
				} else {
					out.Int64Data[i] = sum
				}
			}
		} else if t.DType == Float && other.DType == Int32 {
			for i := range t.Shape[0] {
				sum = 0
				for j := range t.Shape[1] {
					sum += int64(t.FloatData[i*t.Shape[1]+j]) * int64(other.Int32Data[j])
				}
				if out.DType == Int32 {
					out.Int32Data[i] = int32(sum)
				} else {
					out.Int64Data[i] = sum
				}
			}
		} else if t.DType == Float && other.DType == Int64 {
			for i := range t.Shape[0] {
				sum = 0
				for j := range t.Shape[1] {
					sum += int64(t.FloatData[i*t.Shape[1]+j]) * other.Int64Data[j]
				}
				if out.DType == Int32 {
					out.Int32Data[i] = int32(sum)
				} else {
					out.Int64Data[i] = sum
				}
			}
		} else if t.DType == Double && other.DType == Float {
			for i := range t.Shape[0] {
				sum = 0
				for j := range t.Shape[1] {
					sum += int64(t.DoubleData[i*t.Shape[1]+j]) * int64(other.FloatData[j])
				}
				if out.DType == Int32 {
					out.Int32Data[i] = int32(sum)
				} else {
					out.Int64Data[i] = sum
				}
			}
		} else if t.DType == Double && other.DType == Double {
			for i := range t.Shape[0] {
				sum = 0
				for j := range t.Shape[1] {
					sum += int64(t.DoubleData[i*t.Shape[1]+j]) * int64(other.DoubleData[j])
				}
				if out.DType == Int32 {
					out.Int32Data[i] = int32(sum)
				} else {
					out.Int64Data[i] = sum
				}
			}
		} else if t.DType == Double && other.DType == Int32 {
			for i := range t.Shape[0] {
				sum = 0
				for j := range t.Shape[1] {
					sum += int64(t.DoubleData[i*t.Shape[1]+j]) * int64(other.Int32Data[j])
				}
				if out.DType == Int32 {
					out.Int32Data[i] = int32(sum)
				} else {
					out.Int64Data[i] = sum
				}
			}
		} else if t.DType == Double && other.DType == Int64 {
			for i := range t.Shape[0] {
				sum = 0
				for j := range t.Shape[1] {
					sum += int64(t.DoubleData[i*t.Shape[1]+j]) * other.Int64Data[j]
				}
				if out.DType == Int32 {
					out.Int32Data[i] = int32(sum)
				} else {
					out.Int64Data[i] = sum
				}
			}
		} else if t.DType == Int32 && other.DType == Float {
			for i := range t.Shape[0] {
				sum = 0
				for j := range t.Shape[1] {
					sum += int64(t.Int32Data[i*t.Shape[1]+j]) * int64(other.FloatData[j])
				}
				if out.DType == Int32 {
					out.Int32Data[i] = int32(sum)
				} else {
					out.Int64Data[i] = sum
				}
			}
		} else if t.DType == Int32 && other.DType == Double {
			for i := range t.Shape[0] {
				sum = 0
				for j := range t.Shape[1] {
					sum += int64(t.Int32Data[i*t.Shape[1]+j]) * int64(other.DoubleData[j])
				}
				if out.DType == Int32 {
					out.Int32Data[i] = int32(sum)
				} else {
					out.Int64Data[i] = sum
				}
			}
		} else if t.DType == Int32 && other.DType == Int32 {
			for i := range t.Shape[0] {
				sum = 0
				for j := range t.Shape[1] {
					sum += int64(t.Int32Data[i*t.Shape[1]+j]) * int64(other.Int32Data[j])
				}
				if out.DType == Int32 {
					out.Int32Data[i] = int32(sum)
				} else {
					out.Int64Data[i] = sum
				}
			}
		} else if t.DType == Int32 && other.DType == Int64 {
			for i := range t.Shape[0] {
				sum = 0
				for j := range t.Shape[1] {
					sum += int64(t.Int32Data[i*t.Shape[1]+j]) * other.Int64Data[j]
				}
				if out.DType == Int32 {
					out.Int32Data[i] = int32(sum)
				} else {
					out.Int64Data[i] = sum
				}
			}
		} else if t.DType == Int64 && other.DType == Float {
			for i := range t.Shape[0] {
				sum = 0
				for j := range t.Shape[1] {
					sum += t.Int64Data[i*t.Shape[1]+j] * int64(other.FloatData[j])
				}
				if out.DType == Int32 {
					out.Int32Data[i] = int32(sum)
				} else {
					out.Int64Data[i] = sum
				}
			}
		} else if t.DType == Int64 && other.DType == Double {
			for i := range t.Shape[0] {
				sum = 0
				for j := range t.Shape[1] {
					sum += t.Int64Data[i*t.Shape[1]+j] * int64(other.DoubleData[j])
				}
				if out.DType == Int32 {
					out.Int32Data[i] = int32(sum)
				} else {
					out.Int64Data[i] = sum
				}
			}
		} else if t.DType == Int64 && other.DType == Int32 {
			for i := range t.Shape[0] {
				sum = 0
				for j := range t.Shape[1] {
					sum += t.Int64Data[i*t.Shape[1]+j] * int64(other.Int32Data[j])
				}
				if out.DType == Int32 {
					out.Int32Data[i] = int32(sum)
				} else {
					out.Int64Data[i] = sum
				}
			}
		} else if t.DType == Int64 && other.DType == Int64 {
			for i := range t.Shape[0] {
				sum = 0
				for j := range t.Shape[1] {
					sum += t.Int64Data[i*t.Shape[1]+j] * other.Int64Data[j]
				}
				if out.DType == Int32 {
					out.Int32Data[i] = int32(sum)
				} else {
					out.Int64Data[i] = sum
				}
			}
		}
	}
	return out, nil
}

func createOutputTensor(firstDtype, secondDtype DataType, shape []int) *Tensor {
	result := &Tensor{Shape: shape}
	if firstDtype == secondDtype {
		result.DType = firstDtype
	} else {
		if (firstDtype == Float && secondDtype == Double) || (secondDtype == Float && firstDtype == Double) {
			// both float types, set to double
			result.DType = Double
		} else if (firstDtype == Int32 && secondDtype == Int64) || (secondDtype == Int32 && firstDtype == Int64) {
			// both int types, set to long
			result.DType = Int64
		} else if (firstDtype == Int32 && (secondDtype == Float || secondDtype == Double)) ||
			(secondDtype == Int32 && (firstDtype == Float || firstDtype == Double)) ||
			(firstDtype == Int64 && (secondDtype == Float || secondDtype == Double)) ||
			(secondDtype == Int64 && (firstDtype == Float || firstDtype == Double)) {
			// one is a float and the other is an integer, set to double
			result.DType = Double
		}
	}
	result.Alloc()
	return result
}
