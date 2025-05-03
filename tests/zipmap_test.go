package tests

import "testing"

type maptype interface {
	int64 | string
}

func runzipmap[T maptype](t *testing.T, classes []T, mtype string, shape []int, expect_success bool) {
	sg := Test("ZipMap")
	input := []float32{1, 0, 3, 44, 23, 11.3}
	if mtype == "string" {
		sg.addAttribute("classlabels_strings", classes)
	} else if mtype == "int64" {
		sg.addAttribute("classlabels_int64s", classes)
	} else {
		t.Fatalf("invalid type: %s", mtype)
	}

	batch_size := 1
	if len(shape) > 1 {
		batch_size = shape[0]
	}
	var output []map[T]float32
	if expect_success {
		output := make([]map[T]float32, batch_size)
		for i := range batch_size {
			output[i] = make(map[T]float32)
			for j := range classes {
				output[i][classes[j]] = input[i*3+j]
			}
		}
	}
	sg.addInput("X", shape, input)
	sg.addOutput("Y", output)
	sg.errorBound = 0.0000001
	err := sg.Execute(t)
	if !expect_success && err == nil {
		t.Fatal("expected error")
	}
}

func TestZipMapStringFloat(t *testing.T) {
	runzipmap(t, []string{"class1", "class2", "class3"}, "string", []int{2, 3}, true)
}

func TestZipMapIntFloat(t *testing.T) {
	runzipmap(t, []int64{10, 20, 30}, "int64", []int{2, 3}, true)
}

func TestZipMapIntFloat1D(t *testing.T) {
	runzipmap(t, []int64{10, 20, 30, 40, 50, 60}, "int64", []int{6}, true)
}

func TestZipMapStringFloatColMoreThanLabels(t *testing.T) {
	runzipmap(t, []string{"class1", "class2", "class3"}, "string", []int{1, 6}, false)
}

func TestZipMapStringFloatColLessThanLabels(t *testing.T) {
	runzipmap(t, []string{"class1", "class2", "class3"}, "string", []int{3, 2}, false)
}

func TestZipMapIntFloatColMoreThanLabels(t *testing.T) {
	runzipmap(t, []int64{10, 20, 30}, "int64", []int{1, 6}, false)
}

func TestZipMapIntFloatColLessThanLabels(t *testing.T) {
	runzipmap(t, []int64{10, 20, 30}, "int64", []int{3, 2}, false)
}
