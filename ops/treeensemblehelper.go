package ops

import (
	"fmt"
	"math"
	"sort"
	"strings"

	"github.com/systemEng-Learning/go-ml-deployment/ir"
	"github.com/systemEng-Learning/go-ml-deployment/tensor"
)

type TreeEnsembleAttributes struct {
	name                            []string
	Tensors                         map[string]*tensor.Tensor
	Ints                            map[string][]int64
	Strings                         map[string][][]byte
	base_values                     *tensor.Tensor
	base_values_as_tensor           *tensor.Tensor
	nodes_nodeids                   []int64
	nodes_treeids                   []int64
	nodes_featureids                []int64
	nodes_values                    *tensor.Tensor
	nodes_values_as_tensor          *tensor.Tensor
	nodes_hitrates                  *tensor.Tensor
	nodes_hitrates_as_tensor        *tensor.Tensor
	nodes_modes                     [][]byte
	nodes_truenodeids               []int64
	nodes_falsenodeids              []int64
	nodes_missing_value_tracks_true []int64
	class_treeids                   []int64
	class_nodeids                   []int64
	class_ids                       []int64
	class_weights                   *tensor.Tensor
	class_weights_as_tensor         *tensor.Tensor
	classlabels_strings             [][]byte
	classlabels_int64s              []int64
	post_transform                  string
	target_treeids                  []int64
	target_nodeids                  []int64
	target_ids                      []int64
	target_weights                  *tensor.Tensor
	target_weights_as_tensor        *tensor.Tensor
	n_target                        int64
	aggregate_function              []byte
}

func removeDuplicatesAndSort(input []int64) []int64 {
	uniqueMap := make(map[int64]struct{}) // Use a map to track unique values
	var uniqueSlice []int64
	for _, num := range input {
		if _, exists := uniqueMap[num]; !exists {
			uniqueMap[num] = struct{}{}
			uniqueSlice = append(uniqueSlice, num)
		}
	}
	sort.Slice(uniqueSlice, func(i, j int) bool {
		return uniqueSlice[i] < uniqueSlice[j]
	})

	return uniqueSlice
}

type TreeNodeKey struct {
	TreeID int64
	NodeID int64
}

type TreeEnsemble struct {
	Atts      *TreeEnsembleAttributes
	TreeIds   []int64
	RootIndex map[int64]int
	NodeIndex map[TreeNodeKey]int
}

func (t *TreeEnsemble) Init(node *ir.NodeProto) error {
	t.Atts = &TreeEnsembleAttributes{}

	for _, attr := range node.Attribute {
		switch attr.Name {
		case "base_values":
			t.Atts.base_values = &tensor.Tensor{
				Shape:     []int{len(attr.Floats)},
				DType:     tensor.Float,
				FloatData: attr.Floats,
			}
		case "base_values_as_tensor":
			base_tensor, err := tensor.FromTensorProto(attr.T)
			if err != nil {
				return fmt.Errorf("failed to create tensor from base_values_as_tensor: %v", err)
			}
			t.Atts.base_values_as_tensor = base_tensor
		case "nodes_nodeids":
			t.Atts.nodes_nodeids = attr.Ints
		case "nodes_treeids":
			t.Atts.nodes_treeids = attr.Ints
		case "nodes_featureids":
			t.Atts.nodes_featureids = attr.Ints
		case "nodes_values":
			t.Atts.nodes_values = &tensor.Tensor{
				Shape:     []int{len(attr.Floats)},
				DType:     tensor.Float,
				FloatData: attr.Floats,
			}
		case "nodes_values_as_tensor":
			nodes_tensor, err := tensor.FromTensorProto(attr.T)
			if err != nil {
				return fmt.Errorf("failed to create tensor from nodes_values_as_tensor: %v", err)
			}
			t.Atts.nodes_values_as_tensor = nodes_tensor
		case "nodes_hitrates":
			t.Atts.nodes_hitrates = &tensor.Tensor{
				Shape:     []int{len(attr.Floats)},
				DType:     tensor.Float,
				FloatData: attr.Floats,
			}
		case "nodes_hitrates_as_tensor":
			nodes_hitrates_tensor, err := tensor.FromTensorProto(attr.T)
			if err != nil {
				return fmt.Errorf("failed to create tensor from nodes_hitrates_as_tensor: %v", err)
			}
			t.Atts.nodes_hitrates_as_tensor = nodes_hitrates_tensor
		case "nodes_modes":
			t.Atts.nodes_modes = attr.Strings
		case "nodes_truenodeids":
			t.Atts.nodes_truenodeids = attr.Ints
		case "nodes_falsenodeids":
			t.Atts.nodes_falsenodeids = attr.Ints
		case "nodes_missing_value_tracks_true":
			t.Atts.nodes_missing_value_tracks_true = attr.Ints
		case "class_treeids":
			t.Atts.class_treeids = attr.Ints
		case "class_nodeids":
			t.Atts.class_nodeids = attr.Ints
		case "class_ids":
			t.Atts.class_ids = attr.Ints
		case "class_weights":
			t.Atts.class_weights = &tensor.Tensor{
				Shape:     []int{len(attr.Floats)},
				DType:     tensor.Float,
				FloatData: attr.Floats,
			}
		case "class_weights_as_tensor":
			class_weights_tensor, err := tensor.FromTensorProto(attr.T)
			if err != nil {
				return fmt.Errorf("failed to create tensor from class_weights_as_tensor: %v", err)
			}
			t.Atts.class_weights_as_tensor = class_weights_tensor
		case "classlabels_strings":
			t.Atts.classlabels_strings = attr.Strings
		case "classlabels_int64s":
			t.Atts.classlabels_int64s = attr.Ints
		case "post_transform":
			t.Atts.post_transform = string(attr.S)
		case "target_treeids":
			t.Atts.target_treeids = attr.Ints
		case "target_nodeids":
			t.Atts.target_nodeids = attr.Ints
		case "target_ids":
			t.Atts.target_ids = attr.Ints
		case "target_weights":
			t.Atts.target_weights = &tensor.Tensor{
				Shape:     []int{len(attr.Floats)},
				DType:     tensor.Float,
				FloatData: attr.Floats,
			}
		case "target_weights_as_tensor":
			target_weights_tensor, err := tensor.FromTensorProto(attr.T)
			if err != nil {
				return fmt.Errorf("failed to create tensor from target_weights_as_tensor: %v", err)
			}
			t.Atts.target_weights_as_tensor = target_weights_tensor
		case "n_target":
			t.Atts.n_target = attr.I
		case "aggregate_function":
			t.Atts.aggregate_function = attr.Strings[0]
		default:
			return fmt.Errorf("unsupported attribute: %s", attr.Name)
		}
	}

	t.TreeIds = removeDuplicatesAndSort(t.Atts.nodes_treeids)

	t.RootIndex = make(map[int64]int)
	for _, tid := range t.TreeIds {
		t.RootIndex[tid] = len(t.TreeIds)
	}

	for index, tids := range t.Atts.nodes_treeids {
		t.RootIndex[tids] = min(t.RootIndex[tids], index)
	}

	t.NodeIndex = make(map[TreeNodeKey]int)
	for i := 0; i < len(t.Atts.nodes_nodeids); i++ {
		key := TreeNodeKey{
			TreeID: t.Atts.nodes_treeids[i],
			NodeID: t.Atts.nodes_nodeids[i],
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

func (t *TreeEnsemble) LeafIndexTree(X []float32, treeid int64) int {
	// compute the leaf index for one tree
	index := t.RootIndex[treeid]
	for string(t.Atts.nodes_modes[index]) != "LEAF" {
		var r bool

		x := X[t.Atts.nodes_featureids[index]]
		if math.IsNaN(float64(x)) {

			r = t.Atts.nodes_missing_value_tracks_true[index] >= 1

		} else {
			rules := t.Atts.nodes_modes[index]

			th := t.Atts.nodes_values.FloatData[index]
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

		var nid int64
		if r {
			nid = t.Atts.nodes_truenodeids[index]
		} else {
			nid = t.Atts.nodes_falsenodeids[index]
		}
		index = t.NodeIndex[TreeNodeKey{TreeID: treeid, NodeID: nid}]

	}
	return index
}

func (t *TreeEnsemble) LeaveIndexTrees(X *tensor.Tensor) *tensor.Tensor {
	shape := X.Shape
	if len(shape) == 1 {
		shape = []int{1, shape[0]}
	}
	nSamples := shape[0]
	nFeatures := shape[1]
	cols := 0
	outputs := []int64{}
	for i := 0; i < nSamples; i++ {
		startIdx := i * nFeatures
		endIdx := startIdx + nFeatures
		rowData := X.FloatData[startIdx:endIdx]
		leaves := []int64{}
		for _, treeid := range t.TreeIds {
			o := t.LeafIndexTree(rowData, treeid)
			leaves = append(leaves, int64(o))
		}

		if cols == 0 {
			cols = len(leaves)
		}
		cols = len(leaves)
		
		outputs = append(outputs, leaves...)
	}
	
	tensor_output := &tensor.Tensor{
		Shape:     []int{nSamples, cols},
		DType:     tensor.Int64,
		Int64Data: outputs,
	}
	return tensor_output
}
