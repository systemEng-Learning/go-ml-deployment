package main

import (
	"fmt"
	"log"
	"os"
	"strings"

	"github.com/systemEng-Learning/go-ml-deployment/graph"
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
	printModel(model)
	graphProto := model.GetGraph()
	graph := graph.Graph{}
	graph.Init(graphProto)
	f := [][]float32{{4.8, 3.1, 1.6, 0.2}, {5.1, 2.5, 3.0, 1.1}, {4.8, 3.4, 1.6, 0.2}}
	graph.Execute([]any{f})
	graph.Print()
	f = [][]float32{{5.5, 2.5, 4.0, 1.3}, {6.7, 2.5, 5.8, 1.8}, {5.2, 3.4, 1.4, 0.2},
		{5.0, 3.4, 1.6, 0.4}, {6.4, 2.7, 5.3, 1.9}, {5.2, 3.5, 1.5, 0.2}}
	graph.Execute([]any{f})
	graph.Print()
	o := []float32{7.9, 3.8, 6.4, 2.0}
	graph.Execute([]any{o})
	graph.Print()
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
