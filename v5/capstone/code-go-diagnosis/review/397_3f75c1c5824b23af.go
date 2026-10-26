/*
Copyright (c) 2018 <NAME>

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
*/


package imggen

import "image"
import "image/color"
import "image/draw"
import "github.com/tidwall/rtree"
import "github.com/mad-day/imggen/conv"
import "math"
import "fmt"

const div float64 = 0xFFFF

func diff (a,b color.Color) float64 {
	var v [3]float64
	x,y,z,_ := a.RGBA()
	v[0] = float64(x)/div
	v[1] = float64(y)/div
	v[2] = float64(z)/div
	x,y,z,_ = b.RGBA()
	v[0] *= float64(x)/div
	v[1] *= float64(y)/div
	v[2] *= float64(z)/div
	return math.Sqrt(v[0]+v[1]+v[2])
}

type Model struct {
	elem func() conv.ImageItem
	tree *rtree.RTree
	size, maxsize int
}

func NewModel(elem func() conv.ImageItem, maxsize int) *Model {
	m := new(Model)
	m.elem = elem
	m.tree = rtree.New(nil)
	m.size = 0
	m.maxsize = maxsize
	return m
}
func (m *Model) find(item conv.ImageItem) (other conv.ImageItem) {
	m.tree.KNN(item,true,func(si rtree.Item, dst float64) bool {
		other = si.(conv.ImageItem)
		return false
	})
	return
}
func (m *Model) Train (img image.Image, tol float64) {
	e := m.elem()
	r := img.Bounds()
	for p := r.Min ; p.Y < r.Max.Y ; p.Y++ {
		for ; p.X < r.Max.X ; p.X++ {
			e.Decode(img,r,p)
			se := m.find(e)
			rc := img.At(p.X,p.Y)
			if se!=nil {
				pc := se.Value()
				if diff(rc,pc)>tol { se = nil }
			}
			if se==nil {
				e.SetValue(rc)
				m.tree.Insert(e)
				e = m.elem()
			}
		}
		fmt.Println(p)
		p.X = r.Min.X
	}
}
func (m *Model) Generate (img draw.Image) {
	e := m.elem()
	r := img.Bounds()
	for p := r.Min ; p.Y < r.Max.Y ; p.Y++ {
		for ; p.X < r.Max.X ; p.X++ {
			e.Decode(img,r,p)
			se := m.find(e)
			if se!=nil {
				img.Set(p.X,p.Y,se.Value())
			}
		}
		fmt.Println(p)
		p.X = r.Min.X
	}
}

