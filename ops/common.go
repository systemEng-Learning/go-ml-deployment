package ops

import (
	"math"

	"github.com/systemEng-Learning/go-ml-deployment/tensor"
)

type postTransform int

const (
	NONE postTransform = iota
	PROBIT
	LOGISTIC
	SOFTMAX
	SOFTMAX_ZERO
)

var postTransformMap = map[string]postTransform{
	"NONE":         NONE,
	"PROBIT":       PROBIT,
	"LOGISTIC":     LOGISTIC,
	"SOFTMAX":      SOFTMAX,
	"SOFTMAX_ZERO": SOFTMAX_ZERO,
}

func update_scores[T tensor.Float32_64](scores []T, shape []int, post_transform postTransform, add_second_class int, have_space bool) {
	rows := shape[0]
	cols := shape[1]
	end := rows * cols
	if cols > 1 {
		switch post_transform {
		case PROBIT:
			tensor.Probit(scores)
		case LOGISTIC:
			tensor.Logistic(scores)
		case SOFTMAX:
			tensor.SoftMax(scores, shape)
		case SOFTMAX_ZERO:
			tensor.SoftMaxZero(scores, shape)
		}
	} else {
		if post_transform == PROBIT {
			for i := range end {
				scores[i] = T(tensor.ComputeProbit(float64(scores[i])))
			}
		} else if add_second_class >= 0 {
			// in this case we have a buffer that holds 2x scores.
			var update_scores func(score T, index int)
			if add_second_class == 0 || add_second_class == 1 {
				update_scores = func(score T, index int) {
					scores[index] = 1 - score
					scores[index+1] = score
				}
			} else if add_second_class == 2 || add_second_class == 3 {
				if post_transform == LOGISTIC {
					update_scores = func(score T, index int) {
						scores[index] = T(tensor.ComputeLogistic(float64(-score)))
						scores[index+1] = T(tensor.ComputeLogistic(float64(score)))
					}
				} else {
					update_scores = func(score T, index int) {
						scores[index] = -score
						scores[index+1] = score
					}
				}

			} else {
				return
			}

			if have_space {
				// there's a gap between scores, this is relevant for SVM
				j := 0
				for range rows {
					update_scores(scores[j], j)
					j += 2
				}
			} else {
				// in this case, the actual scores are at the start of the buffer, and for each score we need 2 entries.
				// process the scores from the back to the front so we don't need a separate buffer.
				cur_end := end
				final_end := end * 2
				for cur_end > 0 {
					cur_end--
					final_end -= 2
					update_scores(scores[cur_end], final_end)
				}
			}
		}
	}
}

func sigmoid_probability(score, proba, probb float32) float32 {
	val := float64(score*proba + probb)
	return 1 - float32(tensor.ComputeLogistic(val)) // ref: https://github.com/microsoft/onnxruntime/blob/1f4156c6146d04999e5e419df4ae6628e928aaad/onnxruntime/core/providers/cpu/ml/ml_common.h#L262
}

// https://www.csie.ntu.edu.tw/~cjlin/papers/svmprob/svmprob.pdf
func multiclass_probability(classcount int, r, p []float32) {
	safe_int_classcount := classcount
	sized2 := safe_int_classcount * classcount
	Q := make([]float32, sized2)
	Qp := make([]float32, safe_int_classcount)

	classcount_f := float32(classcount)
	eps := float64(0.005 / classcount_f)

	for i := range safe_int_classcount {
		p[i] = 1.0 / classcount_f
		for j := range i {
			Q[i*safe_int_classcount+i] += r[j*safe_int_classcount+i] * r[j*safe_int_classcount+i]
			Q[i*safe_int_classcount+j] = Q[j*safe_int_classcount+i]
		}
		for j := i + 1; j < safe_int_classcount; j++ {
			Q[i*safe_int_classcount+i] += r[j*safe_int_classcount+i] * r[j*safe_int_classcount+i]
			Q[i*safe_int_classcount+j] = -r[j*safe_int_classcount+i] * r[i*safe_int_classcount+j]
		}
	}

	for range 100 {
		pQp := float32(0)
		for i := range safe_int_classcount {
			Qp[i] = 0
			for j := range safe_int_classcount {
				Qp[i] += Q[i*safe_int_classcount+j] * p[j]
			}
			pQp += p[i] * Qp[i]
		}

		max_error := float64(0)
		for i := range safe_int_classcount {
			err := math.Abs(float64(Qp[i] - pQp))
			if err > max_error {
				max_error = err
			}
		}

		if max_error < eps {
			break
		}

		for i := range safe_int_classcount {
			diff := (-Qp[i] + pQp) / Q[i*safe_int_classcount+i]
			p[i] += diff
			pQp = (pQp + diff*(diff*Q[i*safe_int_classcount+i]+2*Qp[i])) / (1 + diff) / (1 + diff)
			for j := range safe_int_classcount {
				Qp[j] = (Qp[j] + diff*Q[i*safe_int_classcount+j]) / (1 + diff)
				p[j] /= (1 + diff)
			}
		}
	}
}
