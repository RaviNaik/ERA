<filename>bool_test.go
package goutils

import (
	"github.com/stretchr/testify/assert"
	"testing"
)

func TestToBool(t *testing.T) {
	ta := assert.New(t)
	ta.True(ToBool("true"))
	ta.True(ToBool("on"))
	ta.True(ToBool("ON"))
	ta.True(ToBool("On"))
	ta.True(ToBool("On"))

	ta.False(ToBool("false"))
	ta.False(ToBool("off"))
	ta.False(ToBool("OFF"))
	ta.False(ToBool("Off"))
	ta.False(ToBool("Off"))

	ta.False(ToBool("whatever"))
}

func TestToBoolDv(t *testing.T) {
	ta := assert.New(t)
	ta.True(ToBoolDv("true", false))
	ta.True(ToBoolDv("on", false))
	ta.True(ToBoolDv("ON", false))
	ta.True(ToBoolDv("On", false))
	ta.True(ToBoolDv("On", true))

	ta.False(ToBoolDv("false", true))
	ta.False(ToBoolDv("off", true))
	ta.False(ToBoolDv("OFF", true))
	ta.False(ToBoolDv("Off", true))
	ta.False(ToBoolDv("Off", false))

	ta.True(ToBoolDv("whatever", true))
	ta.False(ToBoolDv("whatever", false))
}
