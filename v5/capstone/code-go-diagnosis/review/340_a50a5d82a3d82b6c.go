<filename>GameA1/element.go
package GameA1

type Element struct {
	Neighbors map[string]*Element
	Position  *Vector2
	Special   interface{}
}

func BuildBidimensionalElementArray(size *Size, wrap bool) {
	array := make([][]Element, size.X)
	for x := 0; x < size.X; x++ {
		visuals.Array[x] = make([]Element, size.Y)
		for y := 0; y < size.Y; y++ {
			element := &visuals.Array[x][y]
			element.Size = NewVector2(x, y)
			element.Neighbors = make(map[string]*Element)
		}
	}
	for x := 0; x < size.X; x++ {
		for y := 0; y < size.Y; y++ {
			element := &visuals.Array[x][y]
			if x == size.X-1 {
				if wrap {
					element.Neighbors["right"] = &array[0][y]
				} else {
					element.Neighbors["right"] = nil
				}
			} else {
				element.Neighbors["right"] = &array[x+1][y]
			}
			if y == size.Y-1 {
				if wrap {
					element.Neighbors["down"] = &array[x][0]
				} else {
					element.Neighbors = nil
				}
			} else {
				element.Neighbors["down"] = &array[x][y+1]
			}
			if x == 0 {
				if wrap {
					element.Neighbors["left"] = &array[size.X-1][y]
				} else {
					element.Neighbors["left"] = nil
				}
			} else {
				element.Neighbors["left"] = &array[x-1][y]
			}
			if y == 0 {
				if wrap {
					element.Neighbors["up"] = &array[x][size.Y-1]
				} else {
					element.Neighbors["up"] = nil
				}
			} else {
				element.Neighbors["up"] = &array[x][y-1]
			}
		}
	}
	return array
}
