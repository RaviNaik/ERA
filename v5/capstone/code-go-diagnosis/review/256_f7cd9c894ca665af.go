<reponame>helmutkemper/iotmaker.santa_isabel_theater.platform<gh_stars>0
package tween

import "math"

// en: sinusoidal easing out - decelerating to zero velocity
var KEaseOutSine = func(interactionCurrent, interactionTotal, startValue, delta float64) float64 {
	return delta*math.Sin(interactionCurrent/interactionTotal*(math.Pi/2)) + startValue
}
