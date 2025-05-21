package tests

import (
	"fmt"
	"testing"
)

func TestDictVectorizer(t *testing.T) {
	dv := Test("DictVectorizer")
	fmt.Println(dv)
	dv.addAttribute("string_vocabulary", []string{"a", "b", "c", "d"})
	mapinput := []map[string]int64 {
		{"a": 1, "c": 2, "d": 3},
	}
	dv.addInputMap("X", mapinput)

	dv.addOutput("Y", []int64{1, 0,2,3})
	err := dv.Execute(t)
	if err != nil {
		t.Fatalf("error shouldn't exist: %v", err)
	}
}

func TestDictVectorizerString(t *testing.T) {
	dv := Test("DictVectorizer")
	fmt.Println(dv)
	dv.addAttribute("int64_vocabulary", []int64{1, 2, 3, 4})
	mapinput := []map[int64][]byte {
		{1: []byte("a"), 3: []byte("c"), 4: []byte("d")},
	}
	dv.addInputMap("X", mapinput)
	dv.addOutput("Y", []string{"a", "", "c", "d"})
	err := dv.Execute(t)
	if err != nil {
		t.Fatalf("error shouldn't exist: %v", err)
	}
}

func TestDictVectorizer2DString(t *testing.T) {
	dv := Test("DictVectorizer")
	fmt.Println(dv)
	dv.addAttribute("int64_vocabulary", []int64{1, 2, 3, 4})
	mapinput := []map[int64][]byte {
		{1: []byte("a"), 3: []byte("c"), 4: []byte("d")},
		{1: []byte("a"), 2: []byte("c"), 4: []byte("d")},
	}
	dv.addInputMap("X", mapinput)
	dv.addOutput("Y", [][]string{{"a", "", "c", "d"}, {"a", "c", "", "d"}})
	err := dv.Execute(t)
	if err != nil {
		t.Fatalf("error shouldn't exist: %v", err)
	}
}

