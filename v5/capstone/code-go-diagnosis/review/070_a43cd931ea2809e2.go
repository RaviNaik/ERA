<filename>customtarget_test.go
package jlog

import (
	"testing"

	"github.com/jerejones/jlog/target"
)

type testTarget struct {
	writeRx []string
}

func (t *testTarget) Write(entry string) {
	t.writeRx = append(t.writeRx, entry)
}

func TestRegisterCustomLoggerTarget(t *testing.T) {
	tgt := testTarget{}
	RegisterCustomLoggerTarget(t.Name(), &tgt)

	m, err := NewManager(&Config{
		AutoReload: false,
		Routes: []RouterConfig{{
			Name:    t.Name(),
			WriteTo: t.Name(),
		}},
		Targets: []target.Config{{
			Name:   t.Name(),
			Layout: "${level} ${message}",
			Type:   t.Name(),
		}},
	})
	if err != nil {
		t.Fatalf("Err: %v", err)
	}
	lgr := m.GetNamedLogger(t.Name())
	lgr.Info("Test")

	expected := "INFO  Test"
	if expected != tgt.writeRx[0] {
		t.Errorf("Expected %s\nGot %s", expected, tgt.writeRx[0])
	}
}
