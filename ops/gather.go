package ops

import (
	"fmt"

	"github.com/systemEng-Learning/go-ml-deployment/ir"
	"github.com/systemEng-Learning/go-ml-deployment/kernel"
	"github.com/systemEng-Learning/go-ml-deployment/tensor"
)

type Gather struct {
	input   int
	axis    int64
	indices *tensor.Tensor
	output  int
}

func (ga *Gather) Init(k *kernel.Kernel, node *ir.NodeProto) error {
	for _, name := range node.Input {
		indices, _ := k.GetInitializer(name)
		if indices != nil {
			ga.indices = indices
			continue
		}
		input, err := k.RegisterReader(name)
		if err != nil {
			return err
		}
		ga.input = input
	}
	for _, attr := range node.Attribute {
		switch attr.Name {
		case "axis":
			ga.axis = attr.I
		default:
			return fmt.Errorf("%s not supported for %s", attr.Name, node.OpType)
		}
	}
	ga.output = k.RegisterWriter(node.Output[0])
	return nil
}

func (ga *Gather) Compute(k *kernel.Kernel) error {
	data, err := k.Input(ga.input)
	if err != nil {
		return err
	}
	input := data.Tensor
	input_shape := input.Shape
	indices_shape := ga.indices.Shape
	input_rank := len(input_shape)
	axis, err := tensor.HandleNegativeAxis(ga.axis, int64(input_rank))
	if err != nil {
		return err
	}
	output_shape := make([]int, 0, input_rank-1+len(indices_shape))
	for i := range axis {
		output_shape = append(output_shape, input_shape[int(i)])
	}
	output_shape = append(output_shape, indices_shape...)
	for i := axis + 1; i < int64(input_rank); i++ {
		output_shape = append(output_shape, input_shape[int(i)])
	}
	output, err := k.Output(ga.output, output_shape, input.DType)
	if err != nil {
		return err
	}
	block, err := input.SizeFromDimension(int(axis) + 1)
	if err != nil {
		return err
	}
	M, err := input.SizeToDimension(int(axis))
	if err != nil {
		return err
	}
	N := ga.indices.Size()
	data_batch, err := input.SizeFromDimension(int(axis))
	if err != nil {
		return err
	}
	gathered_batch := N * block
	switch input.DType {
	case tensor.Int32:
		if ga.indices.DType == tensor.Int32 {
			gather(ga.indices.Int32Data, input.Int32Data, output.Int32Data, block, M, N, data_batch, gathered_batch, input_shape, axis)
		} else {
			gather(ga.indices.Int64Data, input.Int32Data, output.Int32Data, block, M, N, data_batch, gathered_batch, input_shape, axis)
		}
	case tensor.Int64:
		if ga.indices.DType == tensor.Int32 {
			gather(ga.indices.Int32Data, input.Int64Data, output.Int64Data, block, M, N, data_batch, gathered_batch, input_shape, axis)
		} else {
			gather(ga.indices.Int64Data, input.Int64Data, output.Int64Data, block, M, N, data_batch, gathered_batch, input_shape, axis)
		}
	case tensor.Float:
		if ga.indices.DType == tensor.Int32 {
			gather(ga.indices.Int32Data, input.FloatData, output.FloatData, block, M, N, data_batch, gathered_batch, input_shape, axis)
		} else {
			gather(ga.indices.Int64Data, input.FloatData, output.FloatData, block, M, N, data_batch, gathered_batch, input_shape, axis)
		}
	case tensor.Double:
		if ga.indices.DType == tensor.Int32 {
			gather(ga.indices.Int32Data, input.DoubleData, output.DoubleData, block, M, N, data_batch, gathered_batch, input_shape, axis)
		} else {
			gather(ga.indices.Int64Data, input.DoubleData, output.DoubleData, block, M, N, data_batch, gathered_batch, input_shape, axis)
		}
	case tensor.String:
		if ga.indices.DType == tensor.Int32 {
			gather(ga.indices.Int32Data, input.StringData, output.StringData, block, M, N, data_batch, gathered_batch, input_shape, axis)
		} else {
			gather(ga.indices.Int64Data, input.StringData, output.StringData, block, M, N, data_batch, gathered_batch, input_shape, axis)
		}
	default:
		return fmt.Errorf("gather: invalid tensor data type")
	}

	return nil
}

func gather[T Integer, U AllType](indices []T, src []U, dst []U, block int64, M int64, N int64, data_batch int64, gathered_batch int64, input_shape []int, axis int64) error {
	axis_dim_limit := input_shape[int(axis)]
	for i := range N {
		idx := indices[i]
		if idx < T(-axis_dim_limit) || idx >= T(axis_dim_limit) {
			return fmt.Errorf("gather: indices element out of bounds, idx=%d, must be within the inclusive range [%d, %d]", idx, -axis_dim_limit, axis_dim_limit-1)
		}
	}
	for index := range M * N {
		batch := index / N
		i := index % N
		src_offset_batch := batch * data_batch
		dst_offset_batch := batch * gathered_batch
		idx := indices[i]
		if idx < 0 {
			idx += T(axis_dim_limit)
		}
		src_offset := int64(T(src_offset_batch) + idx*T(block))
		dst_offset := int64(T(dst_offset_batch) + T(i*block))
		copy(dst[dst_offset:dst_offset+block], src[src_offset:src_offset+block])
	}
	return nil
}
