package ops

import (
	"errors"
	"fmt"
	"math"
	"sort"
	"strings"

	"github.com/systemEng-Learning/go-ml-deployment/ir"
	"github.com/systemEng-Learning/go-ml-deployment/tensor"
)

type TreeEnsembleAttributes struct {
	name []string
	Tensors map[string]*tensor.Tensor
	Ints map[string][]int64
	Strings map[string][][]byte
}

func NewTreeEnsembleAttributes() *TreeEnsembleAttributes {
	return &TreeEnsembleAttributes{
		name: []string{},
		Tensors: make(map[string]*tensor.Tensor),
		Ints: make(map[string][]int64),
		Strings: make(map[string][][]byte),
	}
}

// Todo
// the numbers of Tree attribute are not fully known
// once all tree models are implemented, we can remove this Method
// and manually define all attributed in the TreeEnsembleAttributes
func (t *TreeEnsembleAttributes) GetAttr(name string, out interface{}) error {
    if val, ok := t.Ints[name]; ok {
        if ptr, ok := out.(*[]int64); ok {
            *ptr = val
            return nil
        }
        return errors.New("type mismatch: expected *[]int64")
    }

    if val, ok := t.Strings[name]; ok {
        if ptr, ok := out.(*[][]byte); ok {
            *ptr = val
            return nil
        }
        return errors.New("type mismatch: expected *[][]byte")
    }

    if val, ok := t.Tensors[name]; ok {
        if ptr, ok := out.(**tensor.Tensor); ok {
            *ptr = val
            return nil
        }
        return errors.New("type mismatch: expected **tensor.Tensor")
    }

    return errors.New("attribute not found")
}

func removeDuplicatesAndSort(input []int) []int {
	uniqueMap := make(map[int]struct{}) // Use a map to track unique values
	var uniqueSlice []int
	for _, num := range input {
		if _, exists := uniqueMap[num]; !exists {
			uniqueMap[num] = struct{}{}
			uniqueSlice = append(uniqueSlice, num)
		}
	}
	sort.Ints(uniqueSlice)
	return uniqueSlice
}




type TreeNodeKey struct {
    TreeID int
    NodeID int
}

type TreeEnsemble struct {
	Atts *TreeEnsembleAttributes
	TreeIds  []int
	RootIndex map[int]int
	NodeIndex  map[TreeNodeKey]int
}

func (t *TreeEnsemble) Init(node *ir.NodeProto) error{
	t.Atts = NewTreeEnsembleAttributes()
	for _, attr := range node.Attribute {
		has_tensor := false
		if strings.HasSuffix(attr.Name, "_as_tensor") {
			has_tensor = true
			t.Atts.name = append(t.Atts.name, attr.Name)
		}
		switch attr.Type {
		case ir.AttributeProto_INT:
			if has_tensor {
				int_tensor := &tensor.Tensor{
					Shape: []int{len(attr.Ints)},
					DType: tensor.Int64,
					Int64Data: attr.Ints,
				}
				t.Atts.Tensors[attr.Name] = int_tensor
			} else {
				t.Atts.Ints[attr.Name] = attr.Ints
			}
		case ir.AttributeProto_FLOAT:
			float_tensor := tensor.Create1DDoubleTensorFromFloat(attr.Floats)
			t.Atts.Tensors[attr.Name] = float_tensor
		case ir.AttributeProto_STRING:
			t.Atts.Strings[attr.Name] = attr.Strings

		default:
			return fmt.Errorf("unsupported data type: %s for attribute: %s", attr.Type, attr.Name)
			
		}

	}

	nodeTreeIDs := make([]int, 0)

	if  err:= t.Atts.GetAttr("nodes_treeids", &nodeTreeIDs); err != nil {
		return fmt.Errorf("failed to get attribute nodes_treeids: %v", err)
	}

	t.TreeIds = removeDuplicatesAndSort(nodeTreeIDs)

	t.RootIndex = make(map[int]int)
	for _, tid := range t.TreeIds {
		t.RootIndex[tid] = len(t.TreeIds)
	}

	for index, tids := range nodeTreeIDs {
		t.RootIndex[tids] = min(t.RootIndex[tids], index)
	}

	nodeNodeIDs := make([]int, 0)
	if err := t.Atts.GetAttr("nodes_nodeids", &nodeNodeIDs); err != nil {
		return fmt.Errorf("failed to get attribute nodes_nodeids: %v", err)
	}

	t.NodeIndex = make(map[TreeNodeKey]int)
	for i := 0; i < len(nodeTreeIDs); i++ {
		key := TreeNodeKey{
			TreeID: nodeTreeIDs[i],
			NodeID: nodeNodeIDs[i],
		}
		t.NodeIndex[key] = i
	}

	return nil
}

func (t *TreeEnsemble) String() string {
	var sb strings.Builder
	sb.WriteString("TreeEnsemble:\n")
	sb.WriteString(fmt.Sprintf("TreeIds: %v\n", t.TreeIds))
	sb.WriteString(fmt.Sprintf("RootIndex: %v\n", t.RootIndex))
	sb.WriteString(fmt.Sprintf("Attributes: %v\n", t.Atts))
	return sb.String()
}

func (t *TreeEnsemble) LeafIndexTree(X []float64, treeid int) int {
	// compute the leaf index for one tree
	index := t.RootIndex[treeid]
	nodesModes := [][]byte{}
	if err := t.Atts.GetAttr("nodes_modes", &nodesModes); err != nil {
		return -1
	}
	for string(nodesModes[index]) != "LEAF" {
		var r bool
		nodesFeatureIDs := []int{}
		if err := t.Atts.GetAttr("nodes_featureids", &nodesFeatureIDs); err != nil {
			return -1
		}
		x := X[nodesFeatureIDs[index]]
		if math.IsNaN(x) {
			nodes_missing_value_tracks_true := []int{}
			if err := t.Atts.GetAttr("nodes_missing_value_tracks_true", &nodes_missing_value_tracks_true); err != nil {
				return -1
			}
			r = nodes_missing_value_tracks_true[index] >= 1

		} else {
			rules := nodesModes[index]
			nodes_values := []float64{}
			if err := t.Atts.GetAttr("nodes_values", &nodes_values); err != nil {
				return -1
			}
			th := nodes_values[index]
			switch string(rules) {
			case "BRANCH_LEQ":
				r = x <= th
			case "BRANCH_LT":
				r = x < th
			case "BRANCH_EQ":
				r = x == th
			case "BRANCH_NEQ":
				r = x != th
			case "BRANCH_GT":
				r = x > th
			case "BRANCH_GTE":
				r = x >= th
			default:
				return -1
			}
			
		}
		
		var nid int
		if r {
			nodes_truenodeids := []int{}
			if err := t.Atts.GetAttr("nodes_truenodeids", &nodes_truenodeids); err != nil {
				return -1
			}
			nid = nodes_truenodeids[index]
		} else {
			nodes_falsenodeids := []int{}
			if err := t.Atts.GetAttr("nodes_falsenodeids", &nodes_falsenodeids); err != nil {
				return -1
			}
			nid = nodes_falsenodeids[index]
		}
		index = t.NodeIndex[TreeNodeKey{TreeID: treeid, NodeID: nid}]

	}
	return index
}

func (t *TreeEnsemble) LeaveIndexTrees(X *tensor.Tensor) []int {
	shape := X.Shape
	if len(shape) == 1 {
		shape = []int{1, shape[0]}
	}
	nSamples := shape[0]
	nFeatures := shape[1]

	outputs := make([]int, nSamples)
	for i := 0; i < nSamples; i++ {
		startIdx := i * nFeatures
		endIdx := startIdx + nFeatures
		rowData := X.DoubleData[startIdx:endIdx]
		leaves := make([]int, len(t.TreeIds))
		for j, treeid := range t.TreeIds {
			leaves[j] =t.LeafIndexTree(rowData, treeid)
		}
		outputs = append(outputs, leaves...)
	}
	return outputs
}