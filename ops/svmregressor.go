package ops

import (
	"fmt"

	"github.com/systemEng-Learning/go-ml-deployment/ir"
	"github.com/systemEng-Learning/go-ml-deployment/kernel"
	"github.com/systemEng-Learning/go-ml-deployment/tensor"
)

type svmType int

const (
	svmSvc svmType = iota
	svmLinear
)

type SVMRegressor struct {
	base           SVMBase
	input          int
	output         int
	vector_count   int
	feature_count  int
	support_vector *tensor.Tensor
	coefficients   *tensor.Tensor
	temp           *tensor.Tensor
	rho            float32
	one_class      bool
	mode           svmType
}

func (s *SVMRegressor) Init(k *kernel.Kernel, node *ir.NodeProto) error {
	input, err := k.RegisterReader(node.Input[0])
	if err != nil {
		return err
	}

	s.input = input
	s.base = SVMBase{}
	for _, attr := range node.Attribute {
		switch attr.Name {
		case "coefficients":
			s.coefficients = tensor.Create1DDoubleTensorFromFloat(attr.Floats)
		case "kernel_params":
			if len(attr.Floats) != 3 {
				return fmt.Errorf("svmregressor: kernel_params must contain 3 values, only contains %d values", len(attr.Floats))
			}
			s.base.gamma = attr.Floats[0]
			s.base.coef0 = attr.Floats[1]
			s.base.degree = attr.Floats[2]
		case "kernel_type":
			kt := string(attr.S)
			if kt == "POLY" {
				s.base.kernel_type = Poly
			} else if kt == "RBF" {
				s.base.kernel_type = Rbf
			} else if kt == "SIGMOID" {
				s.base.kernel_type = Sigmoid
			} else {
				s.base.kernel_type = Linear
			}
		case "n_supports":
			s.vector_count = int(attr.I)
		case "one_class":
			s.one_class = (attr.I != 0)
		case "rho":
			s.rho = attr.Floats[0]
		case "support_vectors":
			s.support_vector = tensor.Create1DDoubleTensorFromFloat(attr.Floats)
		default:
			return fmt.Errorf("%s not supported for %s", attr.Name, node.OpType)
		}
	}

	if s.vector_count > 0 {
		if s.support_vector.Shape[0]%s.vector_count != 0 {
			return fmt.Errorf("svmregressor: support_size %d should be divisible by vector count %d", s.support_vector.Shape[0], s.vector_count)
		}
		feature_count := s.support_vector.Shape[0] / s.vector_count
		if s.coefficients.Shape[0] != s.vector_count {
			return fmt.Errorf("svmregressor: coefficient length (%d) != vector_count (%d)", s.coefficients.Shape[0], s.vector_count)
		}
		s.support_vector.Shape = []int{s.vector_count, feature_count}
		s.feature_count = feature_count
		s.mode = svmSvc
	} else {
		s.feature_count = s.coefficients.Shape[0]
		s.coefficients.Shape = []int{1, s.feature_count}
		s.mode = svmLinear
		s.base.kernel_type = Linear
	}
	s.output = k.RegisterWriter(node.Output[0])
	return nil
}

func (s *SVMRegressor) Compute(k *kernel.Kernel) error {
	fmt.Println(s.coefficients)
	fmt.Println(s.support_vector)
	data, err := k.Input(s.input)
	if err != nil {
		return err
	}
	input := data.Tensor
	if len(input.Shape) > 2 {
		return fmt.Errorf("svmregressor: invalid shape %v", input.Shape)
	}

	if len(input.Shape) == 1 {
		input.Shape = []int{1, input.Shape[0]}
	}
	num_batches := input.Shape[0]
	num_features := input.Shape[1]
	if num_features != s.feature_count {
		return fmt.Errorf("svmregressor: column length (%d) != expected column length (%d)", num_features, s.feature_count)
	}
	if num_features <= 0 || num_batches <= 0 {
		return fmt.Errorf("svmregressor: illegal num_features (%d) or illegal num_batches (%d)", num_features, num_batches)
	}
	// input: [num_batches, feature_count] where features could be coefficients or support vectors
	// coefficients: [vector_count]
	// support_vectors : [vector_count, feature_count]

	output, err := k.Output(s.output, []int{num_batches, 1}, tensor.Float)
	if err != nil {
		return nil
	}
	if s.mode == svmSvc {
		if s.temp == nil {
			s.temp = &tensor.Tensor{
				Shape: []int{num_batches, s.vector_count},
				DType: tensor.Float,
			}
			s.temp.Alloc()
		} else {
			s.temp.Reuse([]int{num_batches, s.vector_count})
		}
		s.base.batched_kernel_dot(input, s.support_vector, s.temp, 0)
		s.temp.Dot(s.coefficients, output)
		for i := range num_batches {
			output.FloatData[i] = output.FloatData[i] + s.rho
		}
	} else {
		s.base.batched_kernel_dot(input, s.coefficients, output, s.rho)
	}

	if s.one_class {
		for i := range num_batches {
			if output.FloatData[i] > 0 {
				output.FloatData[i] = 1
			} else {
				output.FloatData[i] = -1
			}
		}
	}
	return nil
}
