package ops

import (
	"fmt"
	"math"

	"github.com/systemEng-Learning/go-ml-deployment/ir"
	"github.com/systemEng-Learning/go-ml-deployment/kernel"
	"github.com/systemEng-Learning/go-ml-deployment/tensor"
)

type TreeEnsembleRegressor struct {
	tree    *TreeEnsemble
	input   int
	outputs []int
}

func (t *TreeEnsembleRegressor) Init(k *kernel.Kernel, node *ir.NodeProto) error {
	input, err := k.RegisterReader(node.Input[0])
	if err != nil {
		return err
	}
	t.input = input

	t.tree = &TreeEnsemble{}
	err = t.tree.Init(node)
	if err != nil {
		return err
	}

	t.outputs = make([]int, len(node.Output))
	for i, output := range node.Output {
		t.outputs[i] = k.RegisterWriter(output)
	}

	return nil
}

func (t *TreeEnsembleRegressor) Compute(k *kernel.Kernel) error {
	data, err := k.Input(t.input)
	if err != nil {
		return err
	}
	input := data.Tensor

	// Compute leaf indices for all samples
	leaveIndex := t.tree.LeaveIndexTrees(input)

	nTargets := int(t.tree.Atts.n_targets) // Use n_targets from TreeEnsembleAttributes
	nSamples := leaveIndex.Shape[0]

	// Initialize result tensor
	res := tensor.CreateEmptyTensor([]int{nSamples, nTargets}, tensor.Float)

	// Initialize target index map
	targetIndex := make(map[TreeNodeKey][]int)
	for i := 0; i < len(t.tree.Atts.target_treeids); i++ {
		tid := t.tree.Atts.target_treeids[i]
		nid := t.tree.Atts.target_nodeids[i]
		key := TreeNodeKey{TreeID: tid, NodeID: nid}
		targetIndex[key] = append(targetIndex[key], i)
	}

	// Aggregate results
	numTrees := len(t.tree.Atts.nodes_treeids)
	for i := 0; i < nSamples; i++ {
		start := i * leaveIndex.Shape[1]
		end := start + leaveIndex.Shape[1]
		indices := leaveIndex.Int64Data[start:end]

		for _, idx := range indices {
			treeID := t.tree.Atts.nodes_treeids[idx]
			nodeID := t.tree.Atts.nodes_nodeids[idx]
			key := TreeNodeKey{TreeID: treeID, NodeID: nodeID}

			if its, ok := targetIndex[key]; ok {
				for _, it := range its {
					targetID := t.tree.Atts.target_ids[it]
					if int(targetID) >= nTargets {
						return fmt.Errorf("target id %d is out of bounds for target labels", targetID)
					}
					resIdx := i*nTargets + int(targetID)
					switch t.tree.Atts.aggregate_function {
					case "SUM", "AVERAGE":
						res.FloatData[resIdx] += t.tree.Atts.target_weights.FloatData[it]
					case "MIN":
						if res.FloatData[resIdx] == 0 {
							res.FloatData[resIdx] = float32(math.MaxFloat32)
						}
						res.FloatData[resIdx] = float32(math.Min(float64(res.FloatData[resIdx]), float64(t.tree.Atts.target_weights.FloatData[it])))
					case "MAX":
						if res.FloatData[resIdx] == 0 {
							res.FloatData[resIdx] = float32(-math.MaxFloat32)
						}
						res.FloatData[resIdx] = float32(math.Max(float64(res.FloatData[resIdx]), float64(t.tree.Atts.target_weights.FloatData[it])))
					default:
						return fmt.Errorf("aggregate_function=%s not supported yet", t.tree.Atts.aggregate_function)
					}
				}
			}
		}
	}

	// Handle "AVERAGE" aggregate function
	if t.tree.Atts.aggregate_function == "AVERAGE" {
		for i := 0; i < len(res.FloatData); i++ {
			res.FloatData[i] /= float32(numTrees)
		}
	}

	// Add base values
	if t.tree.Atts.base_values != nil {
		baseValues := t.tree.Atts.base_values.FloatData
		for i := 0; i < nSamples; i++ {
			for j := 0; j < nTargets; j++ {
				res.FloatData[i*nTargets+j] += baseValues[j]
			}
		}
	}

	// Handle post-transform
	switch t.tree.Atts.post_transform {
	case "NONE", "":
		// No transformation needed
	default:
		return fmt.Errorf("post_transform=%s not implemented", t.tree.Atts.post_transform)
	}

	// Write output tensor
	outputTensor, err := k.Output(t.outputs[0], []int{nSamples, nTargets}, tensor.Float)
	if err != nil {
		return err
	}
	copy(outputTensor.FloatData, res.FloatData)

	return nil
}
