package main

import (
	"fmt"
	"log"
	"math"
)

type Graph struct {
	Node   []Node   `json:"node"`
	Name   string   `json:"name"`
	Input  []Input  `json:"input"`
	Output []Output `json:"output"`
}

type Node struct {
	Input      []string    `json:"input"`
	Output     []string    `json:"output"`
	Name       string      `json:"name"`
	OpType     string      `json:"op_type"`
	Domain     string      `json:"domain"`
	Attributes []Attribute `json:"attribute"`
}

type Input struct {
	Name string `json:"name"`
	Type struct {
		Value struct {
			TensorType struct {
				ElemType int `json:"elem_type"`
				Shape    struct {
					Dim []struct {
						Value interface{} `json:"Value"`
					} `json:"dim"`
				} `json:"shape"`
			} `json:"TensorType"`
		} `json:"Value"`
	} `json:"type"`
}

type Output struct {
	Name string `json:"name"`
	Type struct {
		Value struct {
			TensorType struct {
				ElemType int `json:"elem_type"`
				Shape    struct {
					Dim []struct {
						Value interface{} `json:"Value"`
					} `json:"dim"`
				} `json:"shape"`
			} `json:"TensorType"`
		} `json:"Value"`
	} `json:"type"`
}

type Attribute struct {
	Name   string    `json:"name"`
	Type   int       `json:"type"`
	Ints   []int     `json:"ints,omitempty"`
	Floats []float64 `json:"floats,omitempty"`
	S      string    `json:"s,omitempty"`
	I      int       `json:"i,omitempty"`
}

// **Operators**
type LinearClassifier struct {
	Coefficients [][]float64
	Intercepts   []float64
}

func (lr *LinearClassifier) Compute(input []float64) (int, []float64) {
	if len(input) != len(lr.Coefficients[0]) {
		log.Fatalf("LinearClassifier: input dimension (%d) does not match expected (%d)",
			len(input), len(lr.Coefficients[0]))
	}
	logits := make([]float64, len(lr.Intercepts))
	for i := 0; i < len(lr.Intercepts); i++ {
		for j := 0; j < len(input); j++ {
			logits[i] += input[j] * lr.Coefficients[i][j]
		}
		logits[i] += lr.Intercepts[i]
	}
	probabilities := softmax(logits)
	label := argmax(probabilities)
	return label, probabilities
}

type Cast struct{}

func (c *Cast) Compute(label int) int64 {
	return int64(label)
}

type Normalizer struct{}

func (n *Normalizer) Compute(probabilities []float64) []float64 {
	sum := 0.0
	for _, p := range probabilities {
		sum += p
	}
	for i := range probabilities {
		probabilities[i] /= sum
	}
	return probabilities
}

type ZipMap struct {
	ClassLabels []int
}

func (z *ZipMap) Compute(probabilities []float64) map[int]float64 {
	if len(probabilities) != len(z.ClassLabels) {
		log.Fatalf("ZipMap: probabilities length (%d) does not match number of class labels (%d)",
			len(probabilities), len(z.ClassLabels))
	}
	probMap := make(map[int]float64)
	for i, p := range probabilities {
		probMap[z.ClassLabels[i]] = p
	}
	return probMap
}

// **Helper Functions**
func softmax(logits []float64) []float64 {
	sum := 0.0
	for _, logit := range logits {
		sum += math.Exp(logit)
	}
	probabilities := make([]float64, len(logits))
	for i, logit := range logits {
		probabilities[i] = math.Exp(logit) / sum
	}
	return probabilities
}

func argmax(arr []float64) int {
	maxIdx := 0
	maxVal := arr[0]
	for i, val := range arr {
		if val > maxVal {
			maxIdx = i
			maxVal = val
		}
	}
	return maxIdx
}

// Map to hold computed results
var dataStore = make(map[string]interface{})

// **Graph Execution**
func ExecuteGraph(graph Graph, inputData []float64) map[string]interface{} {
	inputDims := graph.Input[0].Type.Value.TensorType.Shape.Dim
	var expectedFeatures int
	if len(inputDims) >= 2 && inputDims[1].Value != nil {
		if dimMap, ok := inputDims[1].Value.(map[string]interface{}); ok {
			if dim, ok := dimMap["DimValue"]; ok {
				fmt.Printf("DimValue: %v\n", dim)
				fmt.Printf("DimValue type: %T\n", dim)
				expectedFeatures = int(dim.(float64))
			}
		}
	}
	if expectedFeatures != 0 && expectedFeatures != len(inputData) {
		log.Fatalf("Input dimension mismatch: expected %d features, got %d", expectedFeatures, len(inputData))
	}

	dataStore[graph.Input[0].Name] = inputData

	for _, node := range graph.Node {
		switch node.OpType {
		case "LinearClassifier":
			coeffFloats := node.Attributes[1].Floats
			intercepts := node.Attributes[2].Floats
			nClasses := len(intercepts)
			if nClasses == 0 {
				log.Fatal("LinearClassifier: no intercepts provided")
			}

			nFeatures := len(coeffFloats) / nClasses

			if len(coeffFloats)%nClasses != 0 {
				log.Fatal("LinearClassifier: coefficients length is not divisible by number of classes")
			}

			if len(inputData) != nFeatures {
				log.Fatalf("LinearClassifier: expected %d features but got %d", nFeatures, len(inputData))
			}

			coeffMatrix := make([][]float64, nClasses)
			for i := 0; i < nClasses; i++ {
				coeffMatrix[i] = coeffFloats[i*nFeatures : (i+1)*nFeatures]
			}
			model := LinearClassifier{Coefficients: coeffMatrix, Intercepts: intercepts}
			label, probabilities := model.Compute(inputData)
			dataStore[node.Output[0]] = label
			dataStore[node.Output[1]] = probabilities

		case "Cast":
			model := Cast{}
			inputLabel, ok := dataStore[node.Input[0]].(int)
			if !ok {
				log.Fatalf("Cast: expected int label from previous node")
			}

			outputLabel := model.Compute(inputLabel)
			dataStore[node.Output[0]] = outputLabel

		case "Normalizer":
			model := Normalizer{}
			probabilities := dataStore[node.Input[0]].([]float64)
			normalizedProb := model.Compute(probabilities)
			dataStore[node.Output[0]] = normalizedProb

		case "ZipMap":
			classLabels := node.Attributes[0].Ints
			model := ZipMap{ClassLabels: classLabels}
			probabilities, ok := dataStore[node.Input[0]].([]float64)
			if !ok {
				log.Fatalf("ZipMap: expected []float64 from previous node")
			}
			if len(probabilities) != len(classLabels) {
				log.Fatalf("ZipMap: probabilities length (%d) does not match number of class labels (%d)",
					len(probabilities), len(classLabels))
			}
			outputProbability := model.Compute(probabilities)
			dataStore[node.Output[0]] = outputProbability

		default:
			log.Fatalf("Unsupported operator: %s", node.OpType)
		}
	}

	outputs := make(map[string]interface{})
	for _, out := range graph.Output {
		val, exists := dataStore[out.Name]
		if !exists {
			log.Fatalf("Expected output %s not found in dataStore", out.Name)
		}
		outputs[out.Name] = val
	}
	return outputs
}

