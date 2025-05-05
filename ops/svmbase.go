package ops

import (
	"math"

	"github.com/systemEng-Learning/go-ml-deployment/tensor"
)

type kernelType int

const (
	Linear kernelType = iota
	Poly
	Sigmoid
	Rbf
)

type SVMBase struct {
	gamma       float32
	coef0       float32
	degree      float32
	kernel_type kernelType
}

func (s *SVMBase) batched_kernel_dot(a *tensor.Tensor, b *tensor.Tensor, out *tensor.Tensor, scalar_c float32) {
	m := a.Shape[0]
	k := a.Shape[1]
	n := b.Shape[0]
	if s.kernel_type == Rbf {
		for batch := range m {
			for support_vector := range n {
				sum := float32(0)
				for feature := range k {
					val := a.FloatData[batch*k+feature] - b.FloatData[support_vector*k+feature]
					sum += val * val
				}
				out.FloatData[batch*n+support_vector] = float32(math.Exp(-float64(s.gamma * sum)))
			}
		}
	} else {
		a.Dot(b, out)
		alpha := float32(1)
		c := scalar_c
		if s.kernel_type != Linear {
			alpha = s.gamma
			c = s.coef0
		}

		length := out.Shape[0] * out.Shape[1]
		for i := range length {
			out.FloatData[i] = out.FloatData[i]*alpha + c
		}
		if s.kernel_type == Poly {
			if s.degree == 2 {
				out.Square()
			} else if s.degree == 3 {
				out.Cube()
			} else {
				out.Power(float64(s.degree))
			}
		} else if s.kernel_type == Sigmoid {
			out.Tanh(out)
		}
	}
}
