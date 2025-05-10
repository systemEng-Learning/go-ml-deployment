package tensor

import (
	"errors"
	"math"
	"reflect"
	"testing"
)

func TestSoftmaxInPlace(t *testing.T) {
	tests := []struct {
		name     string
		tensor   *Tensor
		wantErr  error
		wantData interface{}
	}{
		{
			name: "Valid Float Tensor",
			tensor: &Tensor{
				Shape:     []int{2, 3},
				DType:     Float,
				FloatData: []float32{1.0, 2.0, 3.0, 1.0, 2.0, 3.0},
			},
			wantErr: nil,
			wantData: []float32{
				float32(math.Exp(1-3) / (math.Exp(1-3) + math.Exp(2-3) + math.Exp(3-3))),
				float32(math.Exp(2-3) / (math.Exp(1-3) + math.Exp(2-3) + math.Exp(3-3))),
				float32(math.Exp(3-3) / (math.Exp(1-3) + math.Exp(2-3) + math.Exp(3-3))),
				float32(math.Exp(1-3) / (math.Exp(1-3) + math.Exp(2-3) + math.Exp(3-3))),
				float32(math.Exp(2-3) / (math.Exp(1-3) + math.Exp(2-3) + math.Exp(3-3))),
				float32(math.Exp(3-3) / (math.Exp(1-3) + math.Exp(2-3) + math.Exp(3-3))),
			},
		},
		{
			name: "Valid Double Tensor",
			tensor: &Tensor{
				Shape:      []int{1, 3},
				DType:      Double,
				DoubleData: []float64{1.0, 2.0, 3.0},
			},
			wantErr: nil,
			wantData: []float64{
				math.Exp(1-3) / (math.Exp(1-3) + math.Exp(2-3) + math.Exp(3-3)),
				math.Exp(2-3) / (math.Exp(1-3) + math.Exp(2-3) + math.Exp(3-3)),
				math.Exp(3-3) / (math.Exp(1-3) + math.Exp(2-3) + math.Exp(3-3)),
			},
		},
		{
			name: "Invalid Shape",
			tensor: &Tensor{
				Shape:     []int{3},
				DType:     Float,
				FloatData: []float32{1.0, 2.0, 3.0},
			},
			wantErr: errors.ErrUnsupported,
		},
		{
			name: "Invalid DType",
			tensor: &Tensor{
				Shape:     []int{2, 3},
				DType:     Int32,
				Int32Data: []int32{1, 2, 3, 1, 2, 3},
			},
			wantErr: errors.ErrUnsupported,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			err := tt.tensor.SoftmaxInPlace()
			if !errors.Is(err, tt.wantErr) {
				t.Errorf("SoftmaxInPlace() error = %v, wantErr %v", err, tt.wantErr)
			}

			if tt.wantErr == nil {
				switch tt.tensor.DType {
				case Float:
					if !reflect.DeepEqual(tt.tensor.FloatData, tt.wantData) {
						t.Errorf("SoftmaxInPlace() FloatData = %v, want %v", tt.tensor.FloatData, tt.wantData)
					}
				case Double:
					if !reflect.DeepEqual(tt.tensor.DoubleData, tt.wantData) {
						t.Errorf("SoftmaxInPlace() DoubleData = %v, want %v", tt.tensor.DoubleData, tt.wantData)
					}
				}
			}
		})
	}
}

func TestSoftmaxZeroInPlace(t *testing.T) {
	tests := []struct {
		name     string
		tensor   *Tensor
		wantErr  error
		wantData interface{}
	}{
		{
			name: "Valid Float Tensor",
			tensor: &Tensor{
				Shape:     []int{2, 3},
				DType:     Float,
				FloatData: []float32{1.0, 2.0, 3.0, 1.0, 2.0, 3.0},
			},
			wantErr: nil,
			wantData: []float32{
				float32(math.Exp(1-3) / (math.Exp(1-3) + math.Exp(2-3) + math.Exp(3-3))),
				float32(math.Exp(2-3) / (math.Exp(1-3) + math.Exp(2-3) + math.Exp(3-3))),
				float32(math.Exp(3-3) / (math.Exp(1-3) + math.Exp(2-3) + math.Exp(3-3))),
				float32(math.Exp(1-3) / (math.Exp(1-3) + math.Exp(2-3) + math.Exp(3-3))),
				float32(math.Exp(2-3) / (math.Exp(1-3) + math.Exp(2-3) + math.Exp(3-3))),
				float32(math.Exp(3-3) / (math.Exp(1-3) + math.Exp(2-3) + math.Exp(3-3))),
			},
		},
		{
			name: "Valid Double Tensor",
			tensor: &Tensor{
				Shape:      []int{1, 3},
				DType:      Double,
				DoubleData: []float64{1.0, 2.0, 3.0},
			},
			wantErr: nil,
			wantData: []float64{
				math.Exp(1-3) / (math.Exp(1-3) + math.Exp(2-3) + math.Exp(3-3)),
				math.Exp(2-3) / (math.Exp(1-3) + math.Exp(2-3) + math.Exp(3-3)),
				math.Exp(3-3) / (math.Exp(1-3) + math.Exp(2-3) + math.Exp(3-3)),
			},
		},
		{
			name: "Invalid Shape",
			tensor: &Tensor{
				Shape:     []int{3},
				DType:     Float,
				FloatData: []float32{1.0, 2.0, 3.0},
			},
			wantErr: errors.New("unsupported tensor shape or data type"),
		},
		{
			name: "Invalid DType",
			tensor: &Tensor{
				Shape:     []int{2, 3},
				DType:     Int32,
				Int32Data: []int32{1, 2, 3, 1, 2, 3},
			},
			wantErr: errors.New("unsupported tensor shape or data type"),
		},
		{
			name: "Zero Values in Float Tensor",
			tensor: &Tensor{
				Shape:     []int{1, 3},
				DType:     Float,
				FloatData: []float32{0.0, 0.0, 0.0},
			},
			wantErr:  nil,
			wantData: []float32{0.5, 0.5, 0.5},
		},
		{
			name: "Zero Values in Double Tensor",
			tensor: &Tensor{
				Shape:      []int{1, 3},
				DType:      Double,
				DoubleData: []float64{0.0, 0.0, 0.0},
			},
			wantErr:  nil,
			wantData: []float64{0.5, 0.5, 0.5},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			err := tt.tensor.SoftmaxZeroInPlace()
			if (err != nil || tt.wantErr != nil) && (err == nil || tt.wantErr == nil || err.Error() != tt.wantErr.Error()) {
				t.Errorf("SoftmaxZeroInPlace() error = %v, wantErr %v", err, tt.wantErr)
			}

			if tt.wantErr == nil {
				switch tt.tensor.DType {
				case Float:
					if !reflect.DeepEqual(tt.tensor.FloatData, tt.wantData) {
						t.Errorf("SoftmaxZeroInPlace() FloatData = %v, want %v", tt.tensor.FloatData, tt.wantData)
					}
				case Double:
					if !reflect.DeepEqual(tt.tensor.DoubleData, tt.wantData) {
						t.Errorf("SoftmaxZeroInPlace() DoubleData = %v, want %v", tt.tensor.DoubleData, tt.wantData)
					}
				}
			}
		})
	}
}
