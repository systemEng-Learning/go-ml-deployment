package ops

import "github.com/systemEng-Learning/go-ml-deployment/tensor"

func update_scores[T tensor.Float32_64](scores []T, shape []int, post_transform string, add_second_class int, have_space bool) {
	rows := shape[0]
	cols := shape[1]
	end := rows * cols
	if cols > 1 {
		switch post_transform {
		case "PROBIT":
			tensor.Probit(scores)
		case "LOGISTIC":
			tensor.Logistic(scores)
		case "SOFTMAX":
			tensor.SoftMax(scores, shape)
		case "SOFTMAX_ZERO":
			tensor.SoftMaxZero(scores, shape)
		}
	} else {
		if post_transform == "PROBIT" {
			for i := range end {
				scores[i] = T(tensor.ComputeProbit(float64(scores[i])))
			}
		} else if add_second_class >= 0 {
			// in this case we have a buffer that holds 2x scores.
			var update_scores func(score T, index int)
			if add_second_class == 1 {
				update_scores = func(score T, index int) {
					scores[index] = 1 - score
					scores[index+1] = score
				}
			} else if add_second_class == 2 {
				if post_transform == "LOGISTIC" {
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
