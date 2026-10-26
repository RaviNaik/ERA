<reponame>Fusion/golyze
package main

import (
	"log"
	"os"

	"github.com/fusion/golyz/pkg/data"
	"github.com/fusion/golyz/pkg/deps"
	"github.com/fusion/golyz/pkg/serve"
	"github.com/fusion/golyz/pkg/weight"
	//"github.com/davecgh/go-spew/spew"
)

func main() {
	Dev := false

	l := log.New(os.Stderr, "", 0)

	var chartsData data.ChartsData

	if !Dev {
		chartsData.Weight = weight.BuildWeightData(l)
	}
	chartsData.Deps = deps.BuildDepsData(l)

	serve.RunWebServer(l, &chartsData)

	//renderer := serve.NewFlexRenderer()

	/*
		chartsData.Weight :=  wrap.MapToPieData(l, weight.BuildWeightData(l))
		weightChart := charts.RenderPie(l, weightData)
		renderer.AddCharts(weightChart)

		depsData := deps.BuildDepsData(l)
		depsChart := charts.RenderSankey(l, depsData)
		renderer.AddCharts(depsChart)

		buf := new(bytes.Buffer)
		renderer.Render(buf)
		serve.RunWebServer(l, buf)
	*/
}
