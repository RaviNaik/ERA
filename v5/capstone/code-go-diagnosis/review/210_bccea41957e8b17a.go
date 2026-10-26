<reponame>taqu/goray<gh_stars>0
package core

import (
	"git.maze.io/go/math32"
)

type Sphere struct {
	Center Vector3
	Radius float32
	Material Material
}

func (sphere *Sphere) Hit(ray Ray, tmin float32, tmax float32, record *HitRecord) bool {
	oc := SubVector3(ray.Origin, sphere.Center)
	a := DotVector3(ray.Direction, ray.Direction)
	b := DotVector3(oc, ray.Direction)
	c := DotVector3(oc, oc) - sphere.Radius*sphere.Radius
	discriminant := b*b - a*c
	if discriminant <= 0.0 {
		return false
	}

	inva := 1.0/a
	discriminant = math32.Sqrt(discriminant)
	t := (-b - discriminant)*inva
	if tmin < t && t < tmax {
		record.T = t
		record.Position = ray.PointAt(t)
		record.Normal = DivVector3(SubVector3(record.Position, sphere.Center), sphere.Radius)
		record.Material = sphere.Material
		return true
	}
	t = (-b + discriminant)*inva
	if tmin < t && t < tmax {
		record.T = t
		record.Position = ray.PointAt(t)
		record.Normal = DivVector3(SubVector3(record.Position, sphere.Center), sphere.Radius)
		record.Material = sphere.Material
		return true
	}
	return false
}

