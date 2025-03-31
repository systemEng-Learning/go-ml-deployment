package graph

import (
	"fmt"
	"reflect"
	"slices"

	"github.com/systemEng-Learning/go-ml-deployment/kernel"
	"github.com/systemEng-Learning/go-ml-deployment/tensor"
)

type Number interface {
	int32 | int64 | int | float32 | float64
}

type InputProcessor[T Number] struct {
	index int
	shape []int
	dtype tensor.DataType
}

func (ip *InputProcessor[T]) processStatic(v T, kernel *kernel.Kernel) error {
	if ip.dtype == tensor.StringMap || ip.dtype == tensor.IntMap || ip.dtype == tensor.Undefined {
		return fmt.Errorf("unsupported datatype: %s", ip.dtype)
	}
	shape := slices.Clone(ip.shape)
	if shape[0] == -1 {
		shape[0] = 1
	}
	capacity := shape[0]
	if len(shape) > 1 {
		capacity *= shape[1]
	}

	if capacity != 1 {
		return fmt.Errorf("shape mismatch: static object cannot fit into input expected shape %v", ip.shape)
	}
	t, err := kernel.Output(ip.index, shape, ip.dtype)
	if err != nil {
		return err
	}
	switch ip.dtype {
	case tensor.Float:
		t.FloatData[0] = float32(v)
	case tensor.Double:
		t.DoubleData[0] = float64(v)
	case tensor.Int32:
		t.Int32Data[0] = int32(v)
	case tensor.Int64:
		t.Int64Data[0] = int64(v)
	}
	return nil
}

func (ip *InputProcessor[T]) process1D(v []T, kernel *kernel.Kernel) error {
	if ip.dtype == tensor.StringMap || ip.dtype == tensor.IntMap || ip.dtype == tensor.Undefined {
		return fmt.Errorf("unsupported datatype: %s", ip.dtype)
	}
	shape := slices.Clone(ip.shape)

	switch len(shape) {
	case 1:
		if shape[0] == -1 {
			shape[0] = len(v) // Automatically fit
		} else if len(v) != shape[0] {
			return fmt.Errorf("data of length %d cannot fit expected input of length %d", len(v), shape[0])
		}
	case 2:
		if shape[0] == -1 {
			if len(v)%shape[1] != 0 {
				return fmt.Errorf("data of length %d cannot be reshaped into %v", len(v), shape)
			}
			shape[0] = len(v) / shape[1] // Set row count
		} else if len(v) != shape[0]*shape[1] {
			return fmt.Errorf("data of length %d cannot be reshaped into %v", len(v), shape)
		}
	}

	t, err := kernel.Output(ip.index, shape, ip.dtype)
	if err != nil {
		return err
	}

	switch ip.dtype {
	case tensor.Float:
		for i, val := range v {
			t.FloatData[i] = float32(val)
		}
	case tensor.Double:
		for i, val := range v {
			t.DoubleData[i] = float64(val)
		}
	case tensor.Int32:
		for i, val := range v {
			t.Int32Data[i] = int32(val)
		}
	case tensor.Int64:
		for i, val := range v {
			t.Int64Data[i] = int64(val)
		}
	}
	return nil
}

func (ip *InputProcessor[T]) process2D(v [][]T, kernel *kernel.Kernel) error {
	shape := slices.Clone(ip.shape)
	m := len(v)
	if m == 0 {
		return fmt.Errorf("input is empty")
	}

	n := len(v[0])
	// Ensure all columns sizes are the same
	for i := 1; i < m; i++ {
		if len(v[i]) != n {
			return fmt.Errorf("rows don't have equal length")
		}
	}

	if len(shape) != 2 {
		return fmt.Errorf("input should be 2D, got %dD", len(shape))
	}
	if shape[0] == -1 && n == shape[1] {
		shape[0] = m
	} else if n != shape[1] || (shape[0] > -1 && shape[0] != m) {
		return fmt.Errorf("expected input of shape %v, got [%d, %d]", shape, m, n)
	}
	t, err := kernel.Output(ip.index, shape, ip.dtype)
	if err != nil {
		return err
	}

	switch ip.dtype {
	case tensor.Float:
		for x := range m {
			for y := range n {
				t.FloatData[x*n+y] = float32(v[x][y])
			}
		}
	case tensor.Double:
		for x := range m {
			for y := range n {
				t.DoubleData[x*n+y] = float64(v[x][y])
			}
		}
	case tensor.Int32:
		for x := range m {
			for y := range n {
				t.Int32Data[x*n+y] = int32(v[x][y])
			}
		}
	case tensor.Int64:
		for x := range m {
			for y := range n {
				t.Int64Data[x*n+y] = int64(v[x][y])
			}
		}
	}
	return nil
}

func (g *Graph) setInputs(input []any) error {
	length := len(g.inputs)
	if length != len(input) {
		return fmt.Errorf("the amount of input tensors isn't equal to expected, got %d, wanted %d", len(input), length)
	}
	var err error
	for index := range g.inputs {
		item := input[index]
		shape := g.shapes[index]
		dtype := g.dtypes[index]
		switch item := item.(type) {
		case int32:
			ip := InputProcessor[int32]{index: index, shape: shape, dtype: dtype}
			err = ip.processStatic(item, g.kernel)
		case int:
			ip := InputProcessor[int]{index: index, shape: shape, dtype: dtype}
			err = ip.processStatic(item, g.kernel)
		case int64:
			ip := InputProcessor[int64]{index: index, shape: shape, dtype: dtype}
			err = ip.processStatic(item, g.kernel)
		case float32:
			ip := InputProcessor[float32]{index: index, shape: shape, dtype: dtype}
			err = ip.processStatic(item, g.kernel)
		case float64:
			ip := InputProcessor[float64]{index: index, shape: shape, dtype: dtype}
			err = ip.processStatic(item, g.kernel)
		case []int32:
			ip := InputProcessor[int32]{index: index, shape: shape, dtype: dtype}
			err = ip.process1D(item, g.kernel)
		case []int:
			ip := InputProcessor[int]{index: index, shape: shape, dtype: dtype}
			err = ip.process1D(item, g.kernel)
		case []int64:
			ip := InputProcessor[int64]{index: index, shape: shape, dtype: dtype}
			err = ip.process1D(item, g.kernel)
		case []float32:
			ip := InputProcessor[float32]{index: index, shape: shape, dtype: dtype}
			err = ip.process1D(item, g.kernel)
		case []float64:
			ip := InputProcessor[float64]{index: index, shape: shape, dtype: dtype}
			err = ip.process1D(item, g.kernel)
		case [][]int32:
			ip := InputProcessor[int32]{index: index, shape: shape, dtype: dtype}
			err = ip.process2D(item, g.kernel)
		case [][]int:
			ip := InputProcessor[int]{index: index, shape: shape, dtype: dtype}
			err = ip.process2D(item, g.kernel)
		case [][]int64:
			ip := InputProcessor[int64]{index: index, shape: shape, dtype: dtype}
			err = ip.process2D(item, g.kernel)
		case [][]float32:
			ip := InputProcessor[float32]{index: index, shape: shape, dtype: dtype}
			err = ip.process2D(item, g.kernel)
		case [][]float64:
			ip := InputProcessor[float64]{index: index, shape: shape, dtype: dtype}
			err = ip.process2D(item, g.kernel)
		default:
			return fmt.Errorf("unsupported data type: %v", reflect.TypeOf(item))
		}
		if err != nil {
			return err
		}
	}
	return nil
}
