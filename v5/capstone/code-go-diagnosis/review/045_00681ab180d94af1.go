package phass

import (
	"testing"
)

func TestSkinfoldsSum(t *testing.T) {
	skf := NewSkinfolds(map[int]float64{
		SKFSubscapular: 13.8,
		SKFTriceps:     8.1,
		SKFBiceps:      3.2,
		SKFChest:       4.6,
		SKFMidaxillary: 4.3,
		SKFSuprailiac:  23.4,
		SKFAbdominal:   25.1,
		SKFThigh:       13.2,
		SKFCalf:        7.3,
	})

	skfSum := 103.0
	if !floatEqual(skf.Sum(), skfSum, 0.01) {
		t.Errorf("Summed %.2f, expected %.2f", skf.Sum(), skfSum)
	}

	skfSumAxial := 71.2
	skfAxial := []int{
		SKFSubscapular,
		SKFChest,
		SKFMidaxillary,
		SKFSuprailiac,
		SKFAbdominal,
	}
	if !floatEqual(skf.SumSpecific(skfAxial), skfSumAxial, 0.01) {
		t.Errorf("Summed %.2f, expected %.2f", skf.SumSpecific(skfAxial), skfSumAxial)
	}

	skfSumAppendicular := 31.8
	skfAppendicular := []int{
		SKFTriceps,
		SKFBiceps,
		SKFThigh,
		SKFCalf,
	}
	if !floatEqual(skf.SumSpecific(skfAppendicular), skfSumAppendicular, 0.01) {
		t.Errorf("Summed %.2f, expted %.2f", skf.SumSpecific(skfAppendicular), skfSumAppendicular)
	}
}
