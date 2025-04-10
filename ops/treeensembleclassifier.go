package ops

import (
	"fmt"

	"github.com/systemEng-Learning/go-ml-deployment/ir"
	"github.com/systemEng-Learning/go-ml-deployment/kernel"
	"github.com/systemEng-Learning/go-ml-deployment/tensor"
)

type TreeEnsembleClassifier struct {
	tree    *TreeEnsemble
	input   int
	outputs []int
}

func (t *TreeEnsembleClassifier) Init(k *kernel.Kernel, node *ir.NodeProto) error {
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

func (t *TreeEnsembleClassifier) Compute(k *kernel.Kernel) error {
	data, err := k.Input(t.input)
	if err != nil {
		return err
	}
	input := data.Tensor

	leave_index := t.tree.LeaveIndexTrees(input)
	
	
	len_class_label_int64s := len(t.tree.Atts.classlabels_int64s)
	len_class_label_strings := len(t.tree.Atts.classlabels_strings)

	
	
	var n_classes int
	if len_class_label_int64s > len_class_label_strings {
		n_classes = len_class_label_int64s
	} else {
		n_classes = len_class_label_strings
	}
	n_samples := leave_index.Shape[0]
	
	res, err := tensor.CreateEmptyTensor([]int{n_samples, n_classes}, tensor.Float)
	if err != nil {
		fmt.Errorf("error creating tensor: %v", err)
	}
	if t.tree.Atts.base_values == nil {
		res.FloatData[0] = 0
	} else {
		baseValues := t.tree.Atts.base_values.FloatData
		for i := 0; i < n_samples; i++ {
			copy(res.FloatData[i*n_classes:(i+1)*n_classes], baseValues)
		}
	}
	


	classIndex := make(map[TreeNodeKey][]int)
	for i := 0; i < len(t.tree.Atts.class_treeids); i++ {
		tid := t.tree.Atts.class_treeids[i]
		nid := t.tree.Atts.class_nodeids[i]
		key := TreeNodeKey{TreeID: tid, NodeID: nid}
		classIndex[key] = append(classIndex[key], i)
	}

	
	
	var classWeights []float32
	if t.tree.Atts.class_weights != nil {
		classWeights = t.tree.Atts.class_weights.FloatData
	} else if t.tree.Atts.class_weights_as_tensor != nil {
		classWeights = t.tree.Atts.class_weights_as_tensor.FloatData
	} else {
		fmt.Errorf("class weights are not set")
	}

	classIDs := t.tree.Atts.class_ids
	numTrees := leave_index.Shape[1]
	
	for i := 0; i < n_samples; i++ {
		start := i * leave_index.Shape[1]
		end := start + numTrees
		indices := leave_index.Int64Data[start:end]
		for _, idx := range indices {
			treeID := t.tree.Atts.nodes_treeids[idx]
			nodeID := t.tree.Atts.nodes_nodeids[idx]
			key := TreeNodeKey{TreeID: treeID, NodeID: nodeID}
			if its, ok := classIndex[key]; ok {
				for _, it := range its {
					classID := classIDs[it]
					if int(classID) >= n_classes {
						return fmt.Errorf("class id %d is out of bounds for class labels", classID)
					}
					resIdx := i*n_classes + int(classID)
					res.FloatData[resIdx] += classWeights[it]
				}
			}
		}
	}

	binary := false
	uniqueClassIDs := make(map[int64]struct{})
	for _, classID := range classIDs {
		uniqueClassIDs[classID] = struct{}{}
	}
	binary = len(uniqueClassIDs) == 1
	if binary {
		if n_classes == 1 && (len(t.tree.Atts.classlabels_int64s) == 1 || len(t.tree.Atts.classlabels_strings) == 1) {
			newRes, err := tensor.CreateEmptyTensor([]int{n_samples, 2}, tensor.Float)
			if err != nil {
				return fmt.Errorf("error creating tensor: %v", err)
			}
			for i := 0; i < n_samples; i++ {
				newRes.FloatData[i*2] = res.FloatData[i]
				if t.tree.Atts.post_transform == "NONE" || t.tree.Atts.post_transform == "" || t.tree.Atts.post_transform == "PROBIT" {
					newRes.FloatData[i*2] = 1 - newRes.FloatData[i*2+1]
				} else {
					newRes.FloatData[i*2] = -newRes.FloatData[i*2+1]
				}

			}
		}
	}

	

	switch t.tree.Atts.post_transform {
		case "LOGISITIC":
			err = res.LogisticInPlace()
			if err != nil {
				return fmt.Errorf("error applying logistic: %v", err)
			}
		case "SOFTMAX":
			err = res.SoftmaxInPlace()
			if err != nil {
				return fmt.Errorf("error applying softmax: %v", err)
			}
		case "PROBIT":
			err = res.ProbitInPlace()
			if err != nil {
				return fmt.Errorf("error applying probit: %v", err)
			}
		case "SOFTMAX_ZERO":
			err = res.SoftmaxZeroInPlace()
			if err != nil {
				return fmt.Errorf("error applying softmax_zero: %v", err)
			}
	}
	

	var scoresTensor *tensor.Tensor
	scoresTensor, err = k.Output(t.outputs[1], []int{n_samples, n_classes}, tensor.Float)
	if err != nil {
		return err
	}
	copy(scoresTensor.FloatData, res.FloatData)

	// Determine labels == argmax
	labels := make([]int64, n_samples)
	for i := 0; i < n_samples; i++ {
		start := i * n_classes
		end := start + n_classes
		row := res.FloatData[start:end]
		maxIdx := 0
		maxVal := row[0]
		for j := 1; j < n_classes; j++ {
			if row[j] > maxVal {
				maxIdx = j
				maxVal = row[j]
			}
		}

		labels[i] = int64(maxIdx)
	}

	// Map labels to class labels
	var finalLabelsTensor *tensor.Tensor
	if len(t.tree.Atts.classlabels_int64s) > 0 {
		intLabels := make([]int64, n_samples)
		for i, lbl := range labels {
			if lbl >= int64(len(t.tree.Atts.classlabels_int64s)) {
				return fmt.Errorf("Label index out of range")
			}
			intLabels[i] = t.tree.Atts.classlabels_int64s[lbl]
		}

		finalLabelsTensor, err = k.Output(t.outputs[0], []int{n_samples}, tensor.Int64)
		if err != nil {
			return err
		}
		copy(finalLabelsTensor.Int64Data, intLabels)
	} else if len(t.tree.Atts.classlabels_strings) > 0 {
		strLabels := make([][]byte, n_samples)
		for i, lbl := range labels {
			if lbl >= int64(len(t.tree.Atts.classlabels_strings)) {
				return fmt.Errorf("Label index out of range")
			}
			strLabels[i] = t.tree.Atts.classlabels_strings[lbl]
		}
		finalLabelsTensor, err = k.Output(t.outputs[0], []int{n_samples}, tensor.String)
		if err != nil {
			return err
		}
		copy(finalLabelsTensor.StringData, strLabels)
	} else {
		return fmt.Errorf("No class labels provided")
	}

	return nil

}
