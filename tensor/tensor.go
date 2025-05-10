package tensor

import (
	"fmt"
	"log"
	"slices"
	"strings"

	"github.com/systemEng-Learning/go-ml-deployment/ir"
)

type DataType int

const (
	Undefined DataType = iota
	Float
	Int32
	Int64
	Double
	StringMap
	IntMap
	String
	StringIntMap
	IntStringMap
	IntDoubleMap
	StringDoubleMap

)

var dataTypeMap = map[DataType]string{
	Undefined: "undefined",
	Float:     "float",
	Int32:     "int32",
	Int64:     "int64",
	Double:    "double",
	StringMap: "stringmap",
	IntMap:    "intmap",
	String:    "string",
	StringIntMap:  "stringintmap",
	IntStringMap:  "intstringmap",
	IntDoubleMap:  "intdoublemap",
	StringDoubleMap: "stringdoublemap",
}

func (dt DataType) String() string {
	return dataTypeMap[dt]
}

type Tensor struct {
	Shape      []int
	DType      DataType
	FloatData  []float32
	Int32Data  []int32
	Int64Data  []int64
	DoubleData []float64
	StringData [][]byte
	IntMap     []map[int64]float32
	StringMap  []map[string]float32
	StringIntMap  []map[string]int64
	IntStringMap  []map[int64][]byte
	IntDoubleMap  []map[int64]float64
	StringDoubleMap []map[string]float64
}

func (t *Tensor) Clone() (*Tensor, error) {
	newTensor := Tensor{}
	switch t.DType {
	case Float:
		newTensor.FloatData = slices.Clone(t.FloatData)
	case Int32:
		newTensor.Int32Data = slices.Clone(t.Int32Data)
	case Int64:
		newTensor.Int64Data = slices.Clone(t.Int64Data)
	case Double:
		newTensor.DoubleData = slices.Clone(t.DoubleData)
	case IntMap:
		newTensor.IntMap = slices.Clone(t.IntMap)
	case StringMap:
		newTensor.StringMap = slices.Clone(t.StringMap)
	case String:
		newTensor.StringData = slices.Clone(t.StringData)
	case StringIntMap:
		newTensor.StringIntMap = slices.Clone(t.StringIntMap)
	case IntStringMap:
		newTensor.IntStringMap = slices.Clone(t.IntStringMap)
	case IntDoubleMap:
		newTensor.IntDoubleMap = slices.Clone(t.IntDoubleMap)
	case StringDoubleMap:
		newTensor.StringDoubleMap = slices.Clone(t.StringDoubleMap)
	default:
		return nil, fmt.Errorf("tensor copy: unsupported data type %d", t.DType)
	}
	newTensor.DType = t.DType
	newTensor.Shape = slices.Clone(t.Shape)
	return &newTensor, nil
}

func CreateEmptyTensor(shape []int, dataType DataType) *Tensor {
	t := &Tensor{
		Shape: shape,
		DType: dataType,
	}
	size := shape[0]
	if len(shape) == 2 {
		size *= shape[1]
	}

	switch dataType {
	case Float:
		t.FloatData = make([]float32, size)
	case Int32:
		t.Int32Data = make([]int32, size)
	case Int64:
		t.Int64Data = make([]int64, size)
	case Double:
		t.DoubleData = make([]float64, size)
	case IntMap:
		t.IntMap = make([]map[int64]float32, shape[0])
	case StringMap:
		t.StringMap = make([]map[string]float32, shape[0])
	case String:
		t.StringData = make([][]byte, size)
	case StringIntMap:
		t.StringIntMap = make([]map[string]int64, shape[0])
	case IntStringMap:
		t.IntStringMap = make([]map[int64][]byte, shape[0])
	case IntDoubleMap:
		t.IntDoubleMap = make([]map[int64]float64, shape[0])
	case StringDoubleMap:
		t.StringDoubleMap = make([]map[string]float64, shape[0])
	}

	return t
}

func Create1DFloatTensor(data []float32) *Tensor {
	t := &Tensor{
		Shape:     []int{len(data)},
		FloatData: data,
		DType:     Float,
	}
	return t
}

func Create1DDoubleTensorFromFloat(data []float32) *Tensor {
	t := &Tensor{
		Shape:     []int{len(data)},
		FloatData: data,
		DType:     Float,
	}
	t.Cast(Double)
	return t
}

func CreateOrReuseTensor(t *Tensor, shape []int, dtype DataType) *Tensor {
	if t == nil {
		t = &Tensor{
			Shape: shape,
			DType: dtype,
		}
		t.Alloc()
	} else {
		t.Reuse(shape)
	}
	return t
}

func (t *Tensor) Clear() {
	switch t.DType {
	case Float:
		t.FloatData = nil
	case Int32:
		t.Int32Data = nil
	case Int64:
		t.Int64Data = nil
	case Double:
		t.DoubleData = nil
	case IntMap:
		t.IntMap = nil
	case StringMap:
		t.StringMap = nil
	case String:
		t.StringData = nil
	case StringIntMap:
		t.StringIntMap = nil
	case IntStringMap:
		t.IntStringMap = nil
	case IntDoubleMap:
		t.IntDoubleMap = nil
	case StringDoubleMap:
		t.StringDoubleMap = nil
	}
}

func (t *Tensor) Capacity() int {
	switch t.DType {
	case Float:
		return len(t.FloatData)
	case Int32:
		return len(t.Int32Data)
	case Int64:
		return len(t.Int64Data)
	case Double:
		return len(t.DoubleData)
	case IntMap:
		return len(t.IntMap)
	case StringMap:
		return len(t.StringMap)
	case String:
		return len(t.StringData)
	case StringIntMap:
		return len(t.StringIntMap)
	case IntStringMap:
		return len(t.IntStringMap)
	case IntDoubleMap:
		return len(t.IntDoubleMap)
	case StringDoubleMap:
		return len(t.StringDoubleMap)
	}
	return 0
}

func (t *Tensor) Alloc() {
	capacity := t.Shape[0]
	if len(t.Shape) > 1 {
		capacity *= t.Shape[1]
	}
	switch t.DType {
	case Float:
		t.FloatData = make([]float32, capacity)
	case Int32:
		t.Int32Data = make([]int32, capacity)
	case Int64:
		t.Int64Data = make([]int64, capacity)
	case Double:
		t.DoubleData = make([]float64, capacity)
	case IntMap:
		t.IntMap = make([]map[int64]float32, t.Shape[0])
	case StringMap:
		t.StringMap = make([]map[string]float32, t.Shape[0])
	case String:
		t.StringData = make([][]byte, capacity)
	case StringIntMap:
		t.StringIntMap = make([]map[string]int64, t.Shape[0])
	case IntStringMap:
		t.IntStringMap = make([]map[int64][]byte, t.Shape[0])
	case IntDoubleMap:
		t.IntDoubleMap = make([]map[int64]float64, t.Shape[0])
	case StringDoubleMap:
		t.StringDoubleMap = make([]map[string]float64, t.Shape[0])
	}
}

func (t *Tensor) Reuse(shape []int) {
	capacity := t.Capacity()
	count := shape[0]
	if len(shape) > 1 {
		count *= shape[1]
	}
	t.Shape = shape
	if capacity < count {
		t.Alloc()
	}
}

func (t *Tensor) IsEmpty() bool {
	if len(t.Shape) == 0 {
		return true
	}
	size := 1
	for i := range t.Shape {
		size *= t.Shape[i]
	}
	return size == 0
}

func (t *Tensor) rawData() any {
	switch t.DType {
	case Float:
		return t.FloatData
	case Double:
		return t.DoubleData
	case Int32:
		return t.Int32Data
	case Int64:
		return t.Int64Data
	case String:
		return t.StringData
	default:
		return nil
	}
}

func (t *Tensor) String() string {
	if t.DType == Undefined {
		return "undefined tensor"
	}
	var s strings.Builder
	// Print data
	s.WriteString("Data: ")
	if len(t.Shape) == 1 {
		t.print1D(&s)
	} else {
		t.print2D(&s)
	}

	// Print shape
	s.WriteString("Shape: [")
	for i := range t.Shape {
		fmt.Fprintf(&s, "%d", t.Shape[i])
		if i < len(t.Shape)-1 {
			s.WriteString(", ")
		}
	}

	// Print datatype
	s.WriteString("]\nDatatype: ")
	s.WriteString(t.DType.String())
	return s.String()
}

func (t *Tensor) print1D(s *strings.Builder) {
	if t.DType == IntMap || t.DType == StringMap {
		s.WriteString("[\n")
	} else {
		s.WriteString("[")
	}
	for i := range t.Shape[0] {
		switch t.DType {
		case Float:
			fmt.Fprintf(s, "%f", t.FloatData[i])
		case Int32:
			fmt.Fprintf(s, "%d", t.Int32Data[i])
		case Int64:
			fmt.Fprintf(s, "%d", t.Int64Data[i])
		case Double:
			fmt.Fprintf(s, "%f", t.DoubleData[i])
		case IntMap:
			fmt.Fprintf(s, "%v", t.IntMap[i])
		case StringMap:
			fmt.Fprintf(s, "%v", t.StringMap[i])
		case String:
			s.WriteString(string(t.StringData[i]))
		case StringIntMap:
			fmt.Fprintf(s, "%v", t.StringIntMap[i])
		case IntStringMap:
			fmt.Fprintf(s, "%v", t.IntStringMap[i])
		case IntDoubleMap:
			fmt.Fprintf(s, "%v", t.IntDoubleMap[i])
		case StringDoubleMap:
			fmt.Fprintf(s, "%v", t.StringDoubleMap[i])
		}
		if i < t.Shape[0]-1 {
			if t.DType == IntMap || t.DType == StringMap || t.DType == StringIntMap || t.DType == IntStringMap || t.DType == IntDoubleMap || t.DType == StringDoubleMap {
				s.WriteString(",\n")
			} else {
				s.WriteString(", ")
			}
		}
	}
	if t.DType == IntMap || t.DType == StringMap || t.DType == StringIntMap || t.DType == IntStringMap || t.DType == IntDoubleMap || t.DType == StringDoubleMap {
		s.WriteString("\n]\n")
	} else {
		s.WriteString("]\n")
	}
}

func (t *Tensor) print2D(s *strings.Builder) {
	if t.DType == IntMap || t.DType == StringMap {
		return
	}
	s.WriteString("[\n")
	for i := range t.Shape[0] {
		s.WriteString("\t[")
		m := t.Shape[1]
		for j := range m {
			switch t.DType {
			case Float:
				fmt.Fprintf(s, "%f", t.FloatData[i*m+j])
			case Int32:
				fmt.Fprintf(s, "%d", t.Int32Data[i*m+j])
			case Int64:
				fmt.Fprintf(s, "%d", t.Int64Data[i*m+j])
			case Double:
				fmt.Fprintf(s, "%f", t.DoubleData[i*m+j])
			case String:
				s.WriteString(string(t.StringData[i*m+j]))
			}
			if j < m-1 {
				s.WriteString(", ")
			}
		}
		s.WriteString("],\n")
	}
	s.WriteString("]\n")
}

func OnnxTypeToDtype(elemType int32) DataType {
	elemTypeStr := ir.TensorProto_DataType_name[elemType]
	switch elemTypeStr {
	case "FLOAT":
		return Float
	case "INT32":
		return Int32
	case "INT64":
		return Int64
	case "DOUBLE":
		return Double
	case "STRING":
		return String
	default:
		log.Printf("onnx type %s has not been defined.\n", elemTypeStr)
		return Undefined
	}
}

func FromTensorProto(Tp *ir.TensorProto) (*Tensor, error) {
	dataType := Tp.DataType
	t := &Tensor{}

	t.Shape = make([]int, len(Tp.Dims))
	for i := range Tp.Dims {
		t.Shape[i] = int(Tp.Dims[i])
	}
	elemTypeStr := ir.TensorProto_DataType_name[dataType]
	switch elemTypeStr {
	case "FLOAT":
		t.FloatData = Tp.FloatData
		t.Shape = []int{len(t.FloatData)}
		t.DType = Float
		return t, nil
	case "INT32":
		t.Int32Data = Tp.Int32Data
		t.DType = Int32
		return t, nil
	case "INT64":
		t.Int64Data = Tp.Int64Data
		t.DType = Int64
		return t, nil
	case "DOUBLE":
		t.DoubleData = Tp.DoubleData
		t.DType = Double
		return t, nil
	case "STRING":
		t.StringData = Tp.StringData
		t.DType = String
		return t, nil
	default:
		return nil, fmt.Errorf("tensor copy: unsupported data type %d", t.DType)
	}
}
