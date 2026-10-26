<filename>internal/typename/goreflect/reflectname_test.go
package goreflect_test

import (
	"reflect"

	. "github.com/dogmatiq/configkit/internal/typename/goreflect"
	. "github.com/dogmatiq/configkit/internal/typename/internal/typenametest"
	. "github.com/onsi/ginkgo"
)

var _ = Describe("func NameOf()", func() {
	Declare(
		func(file string) string {
			switch file {
			case "basic":
				return NameOf(reflect.TypeOf(Basic))
			case "named":
				return NameOf(reflect.TypeOf(Named))
			case "pointer":
				return NameOf(reflect.TypeOf(Pointer))
			case "slice":
				return NameOf(reflect.TypeOf(Slice))
			case "array":
				return NameOf(reflect.TypeOf(Array))
			case "map":
				return NameOf(reflect.TypeOf(Map))
			case "channel":
				return NameOf(reflect.TypeOf(Channel))
			case "receive-only-channel":
				return NameOf(reflect.TypeOf(ReceiveOnlyChannel))
			case "send-only-channel":
				return NameOf(reflect.TypeOf(SendOnlyChannel))
			case "iface":
				return NameOf(reflect.TypeOf(&Interface).Elem())
			case "iface-with-single-method":
				return NameOf(reflect.TypeOf(&InterfaceWithSingleMethod).Elem())
			case "iface-with-multiple-methods":
				return NameOf(reflect.TypeOf(&InterfaceWithMultipleMethods).Elem())
			case "struct":
				return NameOf(reflect.TypeOf(Struct))
			case "struct-with-single-field":
				return NameOf(reflect.TypeOf(StructWithSingleField))
			case "struct-with-multiple-fields":
				return NameOf(reflect.TypeOf(StructWithMultipleFields))
			case "struct-with-embedded-type":
				return NameOf(reflect.TypeOf(StructWithEmbeddedType))
			case "nullary":
				return NameOf(reflect.TypeOf(Nullary))
			case "func-with-multiple-input-params":
				return NameOf(reflect.TypeOf(FuncWithMultipleInputParams))
			case "func-with-single-output-param":
				return NameOf(reflect.TypeOf(FuncWithSingleOutputParam))
			case "func-with-multiple-output-params":
				return NameOf(reflect.TypeOf(FuncWithMultipleOutputParams))
			}

			Fail("no type is available for testing with '%s' file")
			return ""
		})
})
