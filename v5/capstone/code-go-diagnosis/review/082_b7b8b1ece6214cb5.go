<reponame>CodeLineage/tool<filename>format_test.go
package tool

import (
	"bytes"
	"testing"
)

func TestStringToBytes(t *testing.T) {
	x := "Hello Gopher!"
	y := StringToBytes(x)
	z := []byte(x)
	if !bytes.Equal(y, z) {
		t.Fail()
	}
	// unexpected fault address 0x55b15c
	// fatal error: fault
	// [signal SIGSEGV: segmentation violation code=0x2 addr=0x55b15c pc=0x507cf8]
	// y[0] = 'A'
}

func TestBytesToString(t *testing.T) {
	x := []byte("Hello Gopher")
	y := BytesToString(x)
	z := string(x)
	if y != z {
		t.Fail()
	}
}

func Benchmark_NormalBytesToString(b *testing.B) {
	x := []byte("Hello Gopher!Hello Gopher!Hello Gopher!Hello Gopher!")
	for i := 0; i < b.N; i++ {
		_ = string(x)
	}
}

func Benchmark_NormalStringToBytes(b *testing.B) {
	x := "Hello Gopher!Hello Gopher!Hello Gopher!Hello Gopher!"
	for i := 0; i < b.N; i++ {
		_ = []byte(x)
	}
}

func Benchmark_BytesToString(b *testing.B) {
	x := []byte("Hello Gopher!Hello Gopher!Hello Gopher!Hello Gopher!")
	for i := 0; i < b.N; i++ {
		_ = BytesToString(x)
	}
}

func Benchmark_StringToBytes(b *testing.B) {
	x := "Hello Gopher!Hello Gopher!Hello Gopher!Hello Gopher!"
	for i := 0; i < b.N; i++ {
		_ = StringToBytes(x)
	}
}
