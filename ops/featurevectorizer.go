package ops

import (
	"fmt"

	"github.com/systemEng-Learning/go-ml-deployment/ir"
	"github.com/systemEng-Learning/go-ml-deployment/kernel"
	"github.com/systemEng-Learning/go-ml-deployment/tensor"
)


type FeatureVectorizer struct {
	input             []int
	outputs           []int
	inputDimensions    []int64

}

func (d *FeatureVectorizer) Init(k *kernel.Kernel, node *ir.NodeProto) error {
	d.input = make([]int, len(node.Input))
	for i, input := range node.Input {
		r_input, err := k.RegisterReader(input)
		if err != nil {
			return err
		}
		d.input[i] = r_input
	}
	
	for _, attr := range node.Attribute {
		switch attr.Name {
		case "input_dimensions":
			d.inputDimensions = attr.Ints
		default:
			return fmt.Errorf("%s not supported for %s", attr.Name, node.OpType)
		}
	}

	d.outputs = make([]int, len(node.Output))

	for i, output := range node.Output {
		d.outputs[i] = k.RegisterWriter(output)
	}
	return nil
}

func (d *FeatureVectorizer) Compute(k *kernel.Kernel) error {

	if len(d.input) != len(d.inputDimensions) {
		return fmt.Errorf("input and input_dimensions length mismatch")
	}

	total:=0
	for _, dim := range d.inputDimensions {
		total += int(dim)
	}

	data , err := k.Input(d.input[0])
	if err != nil {
		return err
	}
	input := data.Tensor
	input_shape := input.Shape[0]
	if len(input.Shape) ==1 {
		input_shape = 1
	}
	res, err := k.Output(d.outputs[0], []int{input_shape, total}, input.DType)
	
	colStart := 0
	for i, input := range d.input {
		data, err := k.Input(input)
		if err != nil {
			return err
		}
		dim := d.inputDimensions[i]
		input_tensor := data.Tensor
		shape := input_tensor.Shape
		if len(shape) == 1 {
			shape = []int{1, shape[0]}
			input_tensor.Shape = shape
		}

		if len(shape) > 2 {
			return fmt.Errorf("input tensor has more than 2 dimensions")
		}
		
		 
		if dim < int64(shape[1]) {
			truncTensor, err := truncateTensorColumns(input_tensor, int(dim))
            if err != nil {
                return err
            }
            shape = []int{shape[0], int(dim)}
            input_tensor = truncTensor
		}


		if err := copyTensorBlock(res, input_tensor, colStart); err != nil {
            return err
        }

		colStart += int(dim)

	}
	return nil
	
}


func truncateTensorColumns(input *tensor.Tensor, dim int) (*tensor.Tensor, error) {
    batch := input.Shape[0]
    cols := input.Shape[1]
    
    newShape := []int{batch, dim}
    out := &tensor.Tensor{
        DType: input.DType,
        Shape: newShape,
    }
    switch input.DType {
    case tensor.Float:
        out.FloatData = make([]float32, batch*dim)
        for i := 0; i < batch; i++ {
            copy(out.FloatData[i*dim:(i+1)*dim], input.FloatData[i*cols:i*cols+dim])
        }
    case tensor.Double:
        out.DoubleData = make([]float64, batch*dim)
        for i := 0; i < batch; i++ {
            copy(out.DoubleData[i*dim:(i+1)*dim], input.DoubleData[i*cols:i*cols+dim])
        }
    case tensor.Int64:
        out.Int64Data = make([]int64, batch*dim)
        for i := 0; i < batch; i++ {
            copy(out.Int64Data[i*dim:(i+1)*dim], input.Int64Data[i*cols:i*cols+dim])
        }
    case tensor.Int32:
        out.Int32Data = make([]int32, batch*dim)
        for i := 0; i < batch; i++ {
            copy(out.Int32Data[i*dim:(i+1)*dim], input.Int32Data[i*cols:i*cols+dim])
        }
    case tensor.String:
        out.StringData = make([][]byte, batch*dim)
        for i := 0; i < batch; i++ {
            copy(out.StringData[i*dim:(i+1)*dim], input.StringData[i*cols:i*cols+dim])
        }
    default:
        return nil, fmt.Errorf("unsupported dtype: %v", input.DType)
    }
    return out, nil
}


func copyTensorBlock(res, input *tensor.Tensor, colStart int) error {
    batch := input.Shape[0]
    inputDim := input.Shape[1]
    resShape := res.Shape
    if len(resShape) != 2 {
        return fmt.Errorf("result tensor must be 2D")
    }
    if batch != resShape[0] {
        return fmt.Errorf("batch size mismatch: input %d, res %d", batch, resShape[0])
    }
    if colStart+inputDim > resShape[1] {
        return fmt.Errorf("input block exceeds result tensor columns")
    }

    switch res.DType {
    case tensor.Float:
        for i := 0; i < batch; i++ {
            for j := 0; j < inputDim; j++ {
                res.FloatData[i*resShape[1]+colStart+j] = input.FloatData[i*inputDim+j]
            }
        }
    case tensor.Double:
        for i := 0; i < batch; i++ {
            for j := 0; j < inputDim; j++ {
                res.DoubleData[i*resShape[1]+colStart+j] = input.DoubleData[i*inputDim+j]
            }
        }
    case tensor.Int64:
        for i := 0; i < batch; i++ {
            for j := 0; j < inputDim; j++ {
                res.Int64Data[i*resShape[1]+colStart+j] = input.Int64Data[i*inputDim+j]
            }
        }
    case tensor.Int32:
        for i := 0; i < batch; i++ {
            for j := 0; j < inputDim; j++ {
                res.Int32Data[i*resShape[1]+colStart+j] = input.Int32Data[i*inputDim+j]
            }
        }
    case tensor.String:
        for i := 0; i < batch; i++ {
            for j := 0; j < inputDim; j++ {
                res.StringData[i*resShape[1]+colStart+j] = input.StringData[i*inputDim+j]
            }
        }
    default:
        return fmt.Errorf("unsupported tensor dtype: %v", res.DType)
    }
    return nil
}