package ops

import (
	"fmt"
	"math"

	"github.com/systemEng-Learning/go-ml-deployment/ir"
	"github.com/systemEng-Learning/go-ml-deployment/kernel"
	"github.com/systemEng-Learning/go-ml-deployment/tensor"
)

type mode int

const (
	Max mode = iota
	L1
	L2
)

type Normalizer struct {
	input  int
	output int
	norm   mode
}

func (n *Normalizer) Init(k *kernel.Kernel, node *ir.NodeProto) error {
	input, err := k.RegisterReader(node.Input[0])
	if err != nil {
		return err
	}
	n.input = input
	n.norm = Max
	for _, attr := range node.Attribute {
		switch attr.Name {
		case "norm":
			norm := string(attr.S)
			if norm == "MAX" {
				n.norm = Max
			} else if norm == "L1" {
				n.norm = L1
			} else if norm == "L2" {
				n.norm = L2
			} else {
				return fmt.Errorf("%s not supported as a norm mode for the Normalizer op", norm)
			}
		default:
			return fmt.Errorf("%s not supported for %s", attr.Name, node.OpType)
		}
	}
	n.output = k.RegisterWriter(node.Output[0])
	return nil
}

func (n *Normalizer) Compute(k *kernel.Kernel) error {
	data, err := k.Input(n.input)
	if err != nil {
		return err
	}
	var input = data.Tensor
	output, err := k.Output(n.output, input.Shape, tensor.Float)
	if err != nil {
		return err
	}
	rows := input.Shape[0]
	cols := rows
	if len(input.Shape) == 1 {
		rows = 1
	} else {
		cols = input.Shape[1]
	}
	if input.DType == tensor.Float {
		normalize(input.FloatData, output.FloatData, rows, cols, n.norm)
	} else if input.DType == tensor.Double {
		normalize(input.DoubleData, output.FloatData, rows, cols, n.norm)
	} else if input.DType == tensor.Int32 {
		normalize(input.Int32Data, output.FloatData, rows, cols, n.norm)
	} else if input.DType == tensor.Int64 {
		normalize(input.Int64Data, output.FloatData, rows, cols, n.norm)
	} else {
		return fmt.Errorf("normalizer op: unsupported input datatype %d", input.DType)
	}
	return err
}

func normalize[T tensor.Numeric](input []T, output []float32, rows, cols int, norm mode) {
	switch norm {
	case Max:
		normalizeMax(input, output, rows, cols)
	case L1:
		normalizeL1(input, output, rows, cols)
	case L2:
		normalizeL2(input, output, rows, cols)
	}
}

func normalizeMax[T tensor.Numeric](input []T, output []float32, rows, cols int) {
	for i := range rows {
		largest := float32(-3.40282e+38) // Gotten from C++ limits header cus golang's limit is a crackhead
		for j := range cols {
			largest = max(largest, float32(input[i*cols+j]))
		}

		if largest != 0 {
			for j := range cols {
				output[i*cols+j] = float32(input[i*cols+j]) / largest
			}
		} else {
			for j := range cols {
				output[i*cols+j] = float32(input[i*cols+j])
			}
		}
	}
}

func normalizeL1[T tensor.Numeric](input []T, output []float32, rows, cols int) {
	for i := range rows {
		sum := 0.0
		for j := range cols {
			sum += math.Abs(float64(input[i*cols+j]))
		}

		if sum != 0.0 {
			sum := float32(sum)
			for j := range cols {
				output[i*cols+j] = float32(input[i*cols+j]) / sum
			}
		} else {
			for j := range cols {
				output[i*cols+j] = float32(input[i*cols+j])
			}
		}
	}
}

func normalizeL2[T tensor.Numeric](input []T, output []float32, rows, cols int) {
	for i := range rows {
		sum := float32(0.0)
		for j := range cols {
			x := float32(input[i*cols+j])
			sq := (x * x)
			output[i*cols+j] = sq
			sum += sq
		}

		if sum != 0.0 {
			for j := range cols {
				in := input[i*cols+j]
				out := output[i*cols+j]
				if in < 0 {
					output[i*cols+j] = float32(math.Sqrt(float64(out/sum)) * -1)
				} else {
					output[i*cols+j] = float32(math.Sqrt(float64(out / sum)))
				}
			}
		} else {
			for j := range cols {
				output[i*cols+j] = float32(input[i*cols+j])
			}
		}
	}
}
