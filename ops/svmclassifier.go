package ops

import (
	"fmt"

	"github.com/systemEng-Learning/go-ml-deployment/ir"
	"github.com/systemEng-Learning/go-ml-deployment/kernel"
	"github.com/systemEng-Learning/go-ml-deployment/tensor"
)

type SVMClassifier struct {
	base                     SVMBase
	input                    int
	vector_count             int
	class_count              int
	feature_count            int
	using_strings            bool
	vectors_per_class        []int64
	starting_vector          []int64
	rho                      []float32
	proba                    *tensor.Tensor
	probb                    *tensor.Tensor
	support_vectors          *tensor.Tensor
	coefficients             *tensor.Tensor
	classlabels              []int64
	classlabels_string       [][]byte
	mode                     svmType
	post_transform           postTransform
	weights_are_all_positive bool
	kernels_data             *tensor.Tensor
	outputs                  []int
}

func (s *SVMClassifier) Init(k *kernel.Kernel, node *ir.NodeProto) error {
	input, err := k.RegisterReader(node.Input[0])
	if err != nil {
		return err
	}

	s.input = input
	s.base = SVMBase{}
	using_strings := false
	s.post_transform = NONE
	for _, attr := range node.Attribute {
		switch attr.Name {
		case "classlabels_ints":
			s.classlabels = attr.Ints
		case "classlabels_strings":
			s.classlabels_string = attr.Strings
			using_strings = true
		case "coefficients":
			s.coefficients = tensor.Create1DFloatTensor(attr.Floats)
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
		case "post_transform":
			s.post_transform = postTransformMap[string(attr.S)]
		case "prob_a":
			s.proba = tensor.Create1DFloatTensor(attr.Floats)
		case "prob_b":
			s.probb = tensor.Create1DFloatTensor(attr.Floats)
		case "rho":
			s.rho = attr.Floats
		case "support_vectors":
			s.support_vectors = tensor.Create1DFloatTensor(attr.Floats)
		case "vectors_per_class":
			s.vectors_per_class = attr.Ints
		default:
			return fmt.Errorf("%s not supported for %s", attr.Name, node.OpType)
		}
	}
	s.using_strings = using_strings

	if s.coefficients == nil || s.coefficients.IsEmpty() {
		return fmt.Errorf("svmclassifier: coefficient attribute cannot be empty")
	}

	if len(s.rho) == 0 {
		return fmt.Errorf("svmclassifier: rho attribute cannot be empty")
	}

	if s.proba != nil && (s.proba.Shape[0] != s.probb.Shape[0]) {
		return fmt.Errorf("svmclassifier: prob_a length (%d) != prob_b length (%d)", s.proba.Shape[0], s.probb.Shape[0])
	}

	if s.classlabels == nil && s.classlabels_string == nil {
		return fmt.Errorf("svmclassifier: one of classlabel_ints and classlabel_strings should be provided")
	}

	s.starting_vector = make([]int64, len(s.vectors_per_class))
	for i := range s.vectors_per_class {
		s.starting_vector[i] = int64(s.vector_count)
		s.vector_count += int(s.vectors_per_class[i])
	}

	if len(s.classlabels) > 0 {
		s.class_count = len(s.classlabels)
	} else if len(s.classlabels_string) > 0 {
		s.class_count = len(s.classlabels_string)
	} else {
		s.class_count = 1
	}

	length := s.coefficients.Shape[0]
	s.weights_are_all_positive = true
	for i := range length {
		if s.coefficients.FloatData[i] < 0 {
			s.weights_are_all_positive = false
			break
		}
	}

	if s.vector_count > 0 {
		if s.support_vectors.Shape[0]%s.vector_count != 0 {
			return fmt.Errorf("svmclassifier: support_size %d should be divisible by vector count %d",
				s.support_vectors.Shape[0], s.vector_count)
		}
		feature_count := s.support_vectors.Shape[0] / s.vector_count
		s.support_vectors.Shape = []int{s.vector_count, feature_count}
		if s.coefficients.Shape[0]%s.vector_count != 0 {
			return fmt.Errorf("svmclassifier: coeffiecients size %d should be divisible by vector count %d",
				s.coefficients.Shape[0], s.vector_count)
		}
		s.coefficients.Shape = []int{s.coefficients.Shape[0] / s.vector_count, s.vector_count}
		s.feature_count = feature_count
		s.mode = svmSvc
	} else {
		if s.coefficients.Shape[0]%s.class_count != 0 {
			return fmt.Errorf("svmclassifier: coefficient length (%d) should be divisible by feature count (%d)", s.coefficients.Shape[0], s.class_count)
		}
		s.feature_count = s.coefficients.Shape[0] / s.class_count
		s.coefficients.Shape = []int{s.class_count, s.feature_count}
		s.mode = svmLinear
		s.base.kernel_type = Linear
	}
	s.outputs = make([]int, len(node.Output))

	for i, output := range node.Output {
		s.outputs[i] = k.RegisterWriter(output)
	}
	return nil
}

type labelType interface {
	int64 | []byte
}

func chooseClass[T labelType](output []T, index int, max_weight float32, max_class int,
	have_proba, weights_are_all_positive bool, class_labels []T, posclass, negclass T) {
	var output_data T
	if len(class_labels) == 2 {
		if !have_proba {
			if weights_are_all_positive && max_weight >= 0.5 {
				output_data = class_labels[1]
			} else if max_weight > 0 && !weights_are_all_positive {
				output_data = class_labels[1]
			} else {
				output_data = class_labels[max_class]
			}
		} else {
			output_data = class_labels[max_class]
		}
	} else if max_weight > 0 {
		output_data = posclass
	} else {
		output_data = negclass
	}
	output[index] = output_data
}

func (s *SVMClassifier) Compute(k *kernel.Kernel) error {
	data, err := k.Input(s.input)
	if err != nil {
		return err
	}
	input := data.Tensor
	if len(input.Shape) > 2 {
		return fmt.Errorf("svmclassifier: invalid shape %v", input.Shape)
	}

	if len(input.Shape) == 1 {
		input.Shape = []int{1, input.Shape[0]}
	}
	num_batches := input.Shape[0]
	num_features := input.Shape[1]
	if num_features != s.feature_count {
		return fmt.Errorf("svmclassifier: column length (%d) != expected column length (%d)", num_features, s.feature_count)
	}
	if num_features <= 0 || num_batches <= 0 {
		return fmt.Errorf("svmclassifier: illegal num_features (%d) or illegal num_batches (%d)", num_features, num_batches)
	}
	input.Cast(tensor.Float)

	// Total number of classifiers comparing pairs between the classes
	num_classifiers := (s.class_count * (s.class_count - 1)) / 2
	class_count_squared := s.class_count * s.class_count
	have_proba := s.proba != nil
	final_scores_per_batch := s.class_count
	if s.mode == svmSvc && !have_proba {
		if s.class_count > 0 {
			final_scores_per_batch = num_classifiers
		} else {
			final_scores_per_batch = 2
		}
	}

	var output_dtype tensor.DataType
	if s.using_strings {
		output_dtype = tensor.String
	} else {
		output_dtype = tensor.Int64
	}
	labels, err := k.Output(s.outputs[0], []int{num_batches}, output_dtype)
	if err != nil {
		return err
	}

	final_scores, err := k.Output(s.outputs[1], []int{num_batches, final_scores_per_batch}, tensor.Float)
	if err != nil {
		return err
	}

	var votes_data []int64
	var classifier_scores_data []float32
	var probsp2_data []float32
	if s.mode == svmSvc && have_proba {
		probsp2_data = make([]float32, num_batches*class_count_squared)
	}

	write_additional_scores := -1
	num_scores_per_batch := s.class_count
	if s.mode == svmSvc && !have_proba {
		num_scores_per_batch = num_classifiers
		if s.class_count <= 2 {
			if s.post_transform == NONE {
				write_additional_scores = 2
			} else {
				write_additional_scores = 0
			}
		}
	}

	if s.mode == svmLinear {
		// combine the coefficients with the input data and apply the kernel type
		// input: [num_batches, feature_count]
		// coefficients: [class_count, feature_count]
		// out: [num_batches, class_count]
		s.base.batched_kernel_dot(input, s.coefficients, final_scores, s.rho[0])

	} else {
		// if we have one classifier, are writing directly to the final buffer,
		// and will add an additional score in the results, leave a space between each classifier score so that
		var num_slots_per_iteration int
		if write_additional_scores >= 0 {
			num_slots_per_iteration = 2
		} else {
			num_slots_per_iteration = num_classifiers
		}

		if have_proba {
			// let's write to an intermediate buffer first
			classifier_scores_data = make([]float32, num_batches*num_classifiers)
		} else {
			// write directly to the final score output.
			classifier_scores_data = final_scores.FloatData
		}
		if s.kernels_data == nil {
			s.kernels_data = &tensor.Tensor{
				Shape: []int{num_batches, s.vector_count},
				DType: tensor.Float,
			}
			s.kernels_data.Alloc()
		} else {
			s.kernels_data.Reuse([]int{num_batches, s.vector_count})
		}
		votes_data = make([]int64, num_batches*s.class_count)

		// combine the input data with the support vectors and apply the kernel type, write output to kernel
		// input: [num_batches, feature_count]
		// support_vectores: [vector_count, feature_count]
		// kernel: [num_batches, vector_count]
		s.base.batched_kernel_dot(input, s.support_vectors, s.kernels_data, 0)
		for n := range num_batches {
			// reduce scores from kernels using coefficients, taking into account the varying number of support vectors

			cur_kernels := s.kernels_data.FloatData[n*s.vector_count:]
			cur_scores := classifier_scores_data[n*num_slots_per_iteration:]
			cur_votes := votes_data[n*s.class_count:]
			scores_iter := 0

			classifier_idx := 0
			for i := range s.class_count - 1 {
				start_index_i := s.starting_vector[i] // start of support vectors for class i
				class_i_support_count := s.vectors_per_class[i]
				i_coeff_row_offset := s.vector_count * i

				for j := i + 1; j < s.class_count; j++ {
					start_index_j := s.starting_vector[j] // start of support vectors for class j
					class_j_support_count := s.vectors_per_class[j]
					j_coeff_row_offset := s.vector_count * (j - 1)

					sum := float64(0)

					val1_index := j_coeff_row_offset + int(start_index_i)
					val2_index := start_index_i
					for range class_i_support_count {
						val1 := s.coefficients.FloatData[val1_index]
						val2 := cur_kernels[val2_index]
						sum += float64(val1 * val2)
						val1_index++
						val2_index++
					}

					val1_index = i_coeff_row_offset + int(start_index_j)
					val2_index = start_index_j
					for range class_j_support_count {
						val1 := s.coefficients.FloatData[val1_index]
						val2 := cur_kernels[val2_index]
						sum += float64(val1 * val2)
						val1_index++
						val2_index++
					}

					sum += float64(s.rho[classifier_idx])
					classifier_idx++

					cur_scores[scores_iter] = float32(sum)
					scores_iter++
					if sum > 0 {
						cur_votes[i]++
					} else {
						cur_votes[j]++
					}
				}
			}
		}
	}

	for n := range num_batches {
		cur_scores := final_scores.FloatData[n*final_scores_per_batch:]

		if s.mode == svmSvc && have_proba {
			probsp2 := probsp2_data[n*class_count_squared:]
			classifier_scores := classifier_scores_data[n*num_classifiers:]

			index := 0
			for i := range s.class_count - 1 {
				p1 := i*s.class_count + i + 1
				p2 := (i+1)*s.class_count + i
				for j := i + 1; j < s.class_count; j++ {
					val1 := sigmoid_probability(classifier_scores[index], s.proba.FloatData[index], s.probb.FloatData[index])
					val2 := max(val1, 1.0e-7)
					val2 = min(val2, 1-1.0e-7)
					probsp2[p1] = val2
					probsp2[p2] = 1 - val2
					p1++
					p2 += s.class_count
					index++
				}
			}

			// expand scores from num_classifiers to class_count_
			multiclass_probability(s.class_count, probsp2, cur_scores)
		}

		max_weight := float32(0)
		maxclass := -1
		if len(votes_data) > 0 {
			votes := votes_data[n*s.class_count:]
			max_votes := votes[0]
			maxclass = 0
			for i := 1; i < s.class_count; i++ {
				if votes[i] > max_votes {
					maxclass = i
					max_votes = votes[i]
				}
			}
		} else {
			max_weight := cur_scores[0]
			maxclass = 0
			for i := 1; i < final_scores_per_batch; i++ {
				if cur_scores[i] > max_weight {
					max_weight = cur_scores[i]
					maxclass = i
				}
			}
		}

		if num_classifiers == 1 { // binary case
			if s.using_strings {
				chooseClass(labels.StringData, n, max_weight, maxclass, have_proba, s.weights_are_all_positive,
					s.classlabels_string, []byte("1"), []byte("0"))
			} else {
				chooseClass(labels.Int64Data, n, max_weight, maxclass, have_proba, s.weights_are_all_positive,
					s.classlabels, 1, 0)
			}
		} else { // multiclass
			if s.using_strings {
				labels.StringData[n] = s.classlabels_string[maxclass]
			} else {
				labels.Int64Data[n] = s.classlabels[maxclass]
			}
		}

		// write the score for this
		update_scores(cur_scores, []int{1, num_scores_per_batch}, s.post_transform,
			write_additional_scores, true)
	}
	return nil
}
