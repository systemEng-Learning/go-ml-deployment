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
	fmt.Println("Version: ", model.GetIrVersion())
	fmt.Println("Producer Name: ", model.GetProducerName())
	fmt.Println("Producer Version: ", model.GetProducerVersion())
	graph := model.GetGraph()
	inputs := graph.GetInput()
	fmt.Println("Inputs------------------------------")
	for _, input := range inputs {
		fmt.Printf("Name: %s, Type: %v\n", input.GetName(), input.GetType())
	}
	fmt.Println("Nodes-------------------------------")
	for _, node := range graph.GetNode() {
		fmt.Printf("Name: %s, Input: %s -> Output: %s\n", node.GetName(), strings.Join(node.GetInput(), " | "),
			strings.Join(node.GetOutput(), " | "))
	}
	outputs := graph.GetOutput()
	fmt.Println("Inputs------------------------------")
	for _, output := range outputs {
		fmt.Printf("Name: %s, Type: %v\n", output.GetName(), output.GetType())
	}
}
