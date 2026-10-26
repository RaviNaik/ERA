<reponame>ludvigalden/go-fragment
package fragment

import (
	"strings"

	"github.com/ludvigalden/go-typemeta"
)

// StructPath is an interface path
type StructPath struct {
	typeMeta     *typemeta.Struct
	fieldIndices []int
	tailFragment Struct
}

var _ Path = (&StructPath{})

// TailFragment returns the tail fragment of the path
func (sp StructPath) TailFragment() Fragment {
	return sp.tailFragment
}

// AssignTailFragment assigns to the tail fragment
func (sp StructPath) AssignTailFragment(v ...interface{}) StructPath {
	if !sp.tailFragment.IsValid() {
		sp.tailFragment = NewEmptyStruct(sp.TailTypeMeta())
	}
	sp.tailFragment = sp.tailFragment.assignv(v...).(Struct)
	return sp
}

// FieldIndices returns the struct field indices of the path
func (sp StructPath) FieldIndices() []int {
	return sp.fieldIndices
}

// TailTypeMeta returns the type meta of the tail of the struct path
func (sp StructPath) TailTypeMeta() *typemeta.Struct {
	tailTypeMeta := sp.typeMeta
	for _, fieldIndex := range sp.FieldIndices() {
		field := tailTypeMeta.EnsureField(fieldIndex)
		tailTypeMeta = typemeta.StructOf(field.TypeMeta)
	}
	return tailTypeMeta
}

// FieldNames returns the struct field names of the path
func (sp StructPath) FieldNames() []string {
	fieldIndices := sp.FieldIndices()
	if fieldIndices == nil {
		return nil
	}
	currentTypeMeta := sp.typeMeta
	fieldNames := []string{}
	for _, fieldIndex := range fieldIndices {
		structField := currentTypeMeta.EnsureField(fieldIndex)
		fieldNames = append(fieldNames, structField.Name)
		currentTypeMeta = typemeta.StructOf(structField.TypeMeta)
	}
	return fieldNames
}

// JSONFieldNames returns the head fragment
func (sp StructPath) JSONFieldNames() []string {
	fieldIndices := sp.FieldIndices()
	if fieldIndices == nil {
		return nil
	}
	currentTypeMeta := sp.typeMeta
	jsonFieldNames := []string{}
	for _, fieldIndex := range fieldIndices {
		structField := currentTypeMeta.EnsureField(fieldIndex)
		if structField.JSONName == "" {
			return nil
		}
		jsonFieldNames = append(jsonFieldNames, structField.JSONName)
		currentTypeMeta = typemeta.StructOf(structField.TypeMeta)
	}
	return jsonFieldNames
}

// ToFragment returns a struct fragment representing the path, including the tail fragment if defined.
func (sp StructPath) ToFragment() Fragment {
	return sp.ToStructFragment()
}

// ToStructFragment returns a struct fragment representing the path, including the tail fragment if defined.
func (sp StructPath) ToStructFragment() Struct {
	fieldIndices := sp.FieldIndices()
	if fieldIndices == nil {
		return Struct{typeMeta: sp.typeMeta, fields: map[int]StructField{}}
	}
	fieldIndicesLen := len(fieldIndices)
	currentFragment := sp.tailFragment
	// we reverse the field path from up to down
	for rangeEnd := fieldIndicesLen - 1; rangeEnd > 0; rangeEnd-- {
		rangeIndices := fieldIndices[0:rangeEnd]
		currentField := typemeta.EnsureStructFieldAt(sp.typeMeta, rangeIndices)
		newFragment := NewStruct(currentField.TypeMeta)
		if !newFragment.IsValid() {
			panic("Cannot create fragment for field " + currentField.String())
		}
		if currentFragment.IsUndefined() {
			newFragment = newFragment.Add(fieldIndices[rangeEnd])
		} else {
			newFragment = newFragment.Set(fieldIndices[rangeEnd], currentFragment)
		}
		currentFragment = newFragment
	}
	return NewStruct(sp.typeMeta).Set(fieldIndices[0], currentFragment)
}

// Expr returns an expression for the interface path
func (sp StructPath) Expr() string {
	fieldNames := sp.FieldNames()
	if fieldNames == nil {
		return ""
	}
	expr := strings.Join(fieldNames, ".")
	if !sp.tailFragment.IsUndefined() {
		tailFragmentExpr := sp.tailFragment.Expr()
		if tailFragmentExpr != "" {
			expr += " " + tailFragmentExpr
		}
	}
	return expr
}

// JSONExpr returns a JSON expression for the interface path
func (sp StructPath) JSONExpr() string {
	jsonFieldNames := sp.JSONFieldNames()
	if jsonFieldNames == nil {
		return ""
	}
	expr := strings.Join(jsonFieldNames, ".")
	if !sp.tailFragment.IsUndefined() {
		tailFragmentJSONExpr := sp.tailFragment.JSONExpr()
		if tailFragmentJSONExpr != "" {
			expr += " " + tailFragmentJSONExpr
		}
	}
	return expr
}

func (sp StructPath) String() string {
	return sp.typeMeta.Name() + "Path(" + sp.Expr() + ")"
}
