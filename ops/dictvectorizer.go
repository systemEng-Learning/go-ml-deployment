package ops

import (
	"fmt"

	"github.com/systemEng-Learning/go-ml-deployment/ir"
	"github.com/systemEng-Learning/go-ml-deployment/kernel"
	"github.com/systemEng-Learning/go-ml-deployment/tensor"
)

type DictVectorizer struct {
	input             int
	outputs           []int
	string_vocabulary [][]byte
	int64_vocabulary  []int64
}

func (d *DictVectorizer) Init(k *kernel.Kernel, node *ir.NodeProto) error {
	input, err := k.RegisterReader(node.Input[0])
	if err != nil {
		return err
	}
	d.input = input

	for _, attr := range node.Attribute {
		switch attr.Name {
		case "string_vocabulary":
			d.string_vocabulary = attr.Strings
		case "int64_vocabulary":
			d.int64_vocabulary = attr.Ints
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

func (d *DictVectorizer) Compute(k *kernel.Kernel) error {
	data, err := k.Input(d.input)
	if err != nil {
		return err
	}
	input := data.Tensor
	if input.Shape[0] > 1 {
		if d.int64_vocabulary == nil && d.string_vocabulary == nil {
			return fmt.Errorf("int64_vocabulary or string_vocabulary must be provided.")
		}
		dictLabels := make(map[interface{}]int)
		if d.int64_vocabulary != nil {
			for i, v := range d.int64_vocabulary {
				dictLabels[v] = i
			}
		}
		if d.string_vocabulary != nil {
			if d.string_vocabulary == nil {
				return fmt.Errorf("int64_vocabulary or string_vocabulary must be provided.")
			}
			for i, v := range d.string_vocabulary {
				dictLabels[string(v)] = i
			}
		}

		var valuesList []any
		var dtype tensor.DataType
		rowsList := []int{}
		colsList := []int{}
		switch input.DType {
		case tensor.IntMap:
			dtype = tensor.Float
			for i, v := range input.IntMap {
				for k, v1 := range v {
					if _, ok := dictLabels[k]; !ok {
						return fmt.Errorf("key %v not found in vocabulary", k)
					}
					rowsList = append(rowsList, i)
					colsList = append(colsList, dictLabels[k])
					valuesList = append(valuesList, v1)
				}
			}
		case tensor.StringMap:
			dtype = tensor.Float
			for i, v := range input.StringMap {
				for k, v1 := range v {
					if _, ok := dictLabels[k]; !ok {
						return fmt.Errorf("key %v not found in vocabulary", k)
					}
					rowsList = append(rowsList, i)
					colsList = append(colsList, dictLabels[k])
					valuesList = append(valuesList, v1)
				}
			}
		case tensor.StringIntMap:
			dtype = tensor.Int64
			for i, v := range input.StringIntMap {
				for k, v1 := range v {
					if _, ok := dictLabels[k]; !ok {
						return fmt.Errorf("key %v not found in vocabulary", k)

					}
					rowsList = append(rowsList, i)
					colsList = append(colsList, dictLabels[k])
					valuesList = append(valuesList, v1)
				}
			}
		case tensor.IntStringMap:
			dtype = tensor.String
			for i, v := range input.IntStringMap {
				for k, v1 := range v {
					if _, ok := dictLabels[k]; !ok {
						return fmt.Errorf("key %v not found in vocabulary", k)
					}
					rowsList = append(rowsList, i)
					colsList = append(colsList, dictLabels[k])
					valuesList = append(valuesList, v1)
				}
			}
		case tensor.IntDoubleMap:
			dtype = tensor.Double
			for i, v := range input.IntDoubleMap {
				for k, v1 := range v {
					if _, ok := dictLabels[k]; !ok {
						return fmt.Errorf("key %v not found in vocabulary", k)
					}
					rowsList = append(rowsList, i)
					colsList = append(colsList, dictLabels[k])
					valuesList = append(valuesList, v1)
				}
			}
		case tensor.StringDoubleMap:
			dtype = tensor.Double
			for i, v := range input.StringDoubleMap {
				for k, v1 := range v {
					if _, ok := dictLabels[k]; !ok {
						return fmt.Errorf("key %v not found in vocabulary", k)
					}
					rowsList = append(rowsList, i)
					colsList = append(colsList, dictLabels[k])
					valuesList = append(valuesList, v1)
				}
			}
		}

		res, err := k.Output(d.outputs[0], []int{input.Shape[0], len(dictLabels)}, dtype)
		if err != nil {
			return err
		}
		
		numCols := len(dictLabels)
		for i, v := range valuesList {
			r := rowsList[i]
			c := colsList[i]
			index := r*numCols + c
			switch dtype {
			case tensor.Float:
				res.FloatData[index] = v.(float32)
			case tensor.String:
				res.StringData[index] = v.([]byte)
			case tensor.Int64:
				res.Int64Data[index] = v.(int64)
			case tensor.Double:
				res.DoubleData[index] = v.(float64)
			}
		}
	} else if input.Shape[0] == 1 {
		if d.int64_vocabulary == nil && d.string_vocabulary == nil {
			return fmt.Errorf("int64_vocabulary or string_vocabulary must be provided.")
		}
		if d.int64_vocabulary != nil {
			var res *tensor.Tensor
			var err error

			switch input.DType {
			case tensor.IntMap:
				res, err = k.Output(d.outputs[0], []int{ len(d.int64_vocabulary)}, tensor.Float)
				if err != nil {
					return err
				}
				for i, v := range d.int64_vocabulary {
					if _, ok := input.IntMap[0][v]; !ok {
						res.FloatData[i] = 0
					} else {
						res.FloatData[i] = input.IntMap[0][v]
					}
				}
			case tensor.IntStringMap:
				res, err = k.Output(d.outputs[0], []int{ len(d.int64_vocabulary)}, tensor.String)
				if err != nil {
					return err
				}
				for i, v := range d.int64_vocabulary {
					if _, ok := input.IntStringMap[0][v]; !ok {
						res.StringData[i] = []byte("")
					} else {
						res.StringData[i] = input.IntStringMap[0][v]
						
					}
				}
			case tensor.IntDoubleMap:
				res, err = k.Output(d.outputs[0], []int{ len(d.int64_vocabulary)}, tensor.Double)
				if err != nil {
					return err
				}
				for i, v := range d.int64_vocabulary {
					if _, ok := input.IntDoubleMap[0][v]; !ok {
						res.DoubleData[i] = 0
					} else {
						res.DoubleData[i] = input.IntDoubleMap[0][v]
					}
				}
			default:
				return fmt.Errorf("input type not supported")
			}

		} else if d.string_vocabulary != nil {
			var res *tensor.Tensor
			var err error

			switch input.DType {
			case tensor.StringMap:
				res, err = k.Output(d.outputs[0], []int{1, len(d.string_vocabulary)}, tensor.Float)
				if err != nil {
					return err
				}
				for i, v := range d.string_vocabulary {
					vstr := string(v)
					if _, ok := input.StringMap[0][vstr]; !ok {
						res.FloatData[i] = 0
					} else {
						res.FloatData[i] = input.StringMap[0][vstr]
					}
				}

			case tensor.StringDoubleMap:
				res, err = k.Output(d.outputs[0], []int{1, len(d.string_vocabulary)}, tensor.Double)
				if err != nil {
					return err
				}
				for i, v := range d.string_vocabulary {
					vstr := string(v)
					if _, ok := input.StringDoubleMap[0][vstr]; !ok {
						res.DoubleData[i] = 0
					} else {
						res.DoubleData[i] = input.StringDoubleMap[0][vstr]
					}
				}

			case tensor.StringIntMap:
				res, err = k.Output(d.outputs[0], []int{len(d.string_vocabulary)}, tensor.Int64)
				if err != nil {
					return err
				}
				for i, v := range d.string_vocabulary {
					vstr := string(v)
					if _, ok := input.StringIntMap[0][vstr]; !ok {
						res.Int64Data[i] = 0
					} else {
						res.Int64Data[i] = input.StringIntMap[0][vstr]
					}
				}

			default:
				return fmt.Errorf("input type not supported")
			}
		}

	} else {
		return fmt.Errorf("input shape not supported")
	}

	return nil
}
