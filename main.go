package main

import (
	"fmt"
	"log"
	"os"
	"strings"

	"github.com/systemEng-Learning/go-ml-deployment/ir"
	"google.golang.org/protobuf/proto"
)

func main() {
	in, err := os.ReadFile("examples/irislog.onnx")

	if err != nil {
		log.Fatalln("Error reading file:", in)
	}

	model := &ir.ModelProto{}
	if err := proto.Unmarshal(in, model); err != nil {
		log.Fatalln("Failed to parse model file:", err)
	}
	fmt.Println(model)
	printModel(model)
	graphProto := model.GetGraph()
	graph := Graph{graph: graphProto}
	graph.Init()
	f := [][]float32{{4.8, 3.1, 1.6, 0.2}, {5.1, 2.5, 3.0, 1.1}, {4.8, 3.4, 1.6, 0.2}}
	graph.Execute(f)
	graph.PrintOutput()
}

func printModel(model *ir.ModelProto) {
	fmt.Println("Version: ", model.GetIrVersion())
	fmt.Println("Producer Name: ", model.GetProducerName())
	fmt.Println("Producer Version: ", model.GetProducerVersion())
	graph := model.GetGraph()
	fmt.Println("Nodes-------------------------------")
	for _, node := range graph.GetNode() {
		fmt.Printf("Name: %s, Input: %s -> Output: %s\n", node.GetOpType(), strings.Join(node.GetInput(), " | "),
			strings.Join(node.GetOutput(), " | "))
	}
}

func DecipherType(d *ir.TypeProto) {
	v := d.GetValue()
	switch t := v.(type) {
	case *ir.TypeProto_TensorType:
		x := t.TensorType
		fmt.Printf("%s\t", ir.TensorProto_DataType_name[x.ElemType])
		GetShape(x.Shape)
	case *ir.TypeProto_MapType:
		m := t.MapType
		fmt.Printf("Key: %s\n", ir.TensorProto_DataType_name[m.KeyType])
		DecipherType(m.ValueType)
	case *ir.TypeProto_SequenceType:
		s := t.SequenceType
		DecipherType(s.ElemType)
	default:
		fmt.Println("Not supported")
	}
}
