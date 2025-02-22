package main

import (
	"encoding/json"
	"fmt"
	"log"
	"os"

	"google.golang.org/protobuf/proto"

	pb "github.com/systemEng-Learning/go-ml-deployment/onnx"
)

func main() {
	// Load ONNX file
	data, err := os.ReadFile("linearreg.onnx")
	if err != nil {
		log.Fatalf("Failed to open ONNX file: %v", err)
	}

	model := &pb.ModelProto{}
	err = proto.Unmarshal(data, model)
	if err != nil {
		log.Fatalf("Failed to parse ONNX file: %v", err)
	}

	modelJSON, err := json.MarshalIndent(model, "", "  ")
	if err != nil {
		log.Fatalf("Failed to convert model to JSON: %v", err)
	}

	var modelData map[string]interface{}
	err = json.Unmarshal(modelJSON, &modelData)
	if err != nil {
		log.Fatalf("Failed to unmarshal JSON: %v", err)
	}

	if graphData, found := modelData["graph"]; found {
		var graph Graph
		graphBytes, err := json.Marshal(graphData)
		if err != nil {
			log.Fatalf("Failed to marshal graphData: %v", err)
		}
		err = json.Unmarshal(graphBytes, &graph)
		if err != nil {
			log.Fatalf("Failed to unmarshal graph into Graph: %v", err)
		}

		c := ExecuteGraph(graph, []float64{4.8, 3.1, 1.6, 0.2})
		fmt.Println(c)
	} else {
		log.Println("No graph found in the ONNX model")
	}
}
