<reponame>cernops/golbclient
package ci

import (
	"strings"
	"testing"

	logger "github.com/sirupsen/logrus"
	"gitlab.cern.ch/lb-experts/golbclient/lbconfig/utils/runner"
)

// TestCILemonCLI : checks if the alternative [lemon-cli] used in the CI pipeline is OK
func TestCILemonCLI(t *testing.T) {
	logger.SetLevel(logger.ErrorLevel)
	output, err, stderr := runner.Run("/usr/sbin/lemon-cli",
		true, defaultTimeout, "--script", "-m", "13163")
	if err != nil {
		t.Fatal("An error was detected when running the CI [lemon-cli]")
	} else if len(strings.TrimSpace(output)) == 0 {
		t.Fatal("The CI [lemon-cli] failed to return a row value for a pre-defined metric")
	}
	logger.Tracef("CI [lemon-cli] output [%s] Stderr[%s]", output, stderr)
}

func TestLemon(t *testing.T) {
	myTests := []lbTest{
		{title: "CollectdFunctionality",
			configuration: "../test/lbclient_lemon_check_single.conf", expectedMetricValue: 12},
		{title: "ConfigurationFile",
			configuration: "../test/lbclient_lemon_check.conf", expectedMetricValue: 8},
		{title: "LemonFailed",
			configuration: "../test/lbclient_lemon_check_fail.conf", expectedMetricValue: -12},
	}

	runMultipleTests(t, myTests)
}

func TestLemonLoad(t *testing.T) {
	myTests := []lbTest{
		{title: "LemonLoadSingle",
			configuration: "../test/lbclient_lemon_load_single.conf", expectedMetricValue: 1},
		{title: "LemonLoad",
			configuration: "../test/lbclient_lemon_load.conf", expectedMetricValue: 50},
		{title: "LemonFailed",
			configuration: "../test/lbclient_lemon_check_fail.conf", expectedMetricValue: -12}}

	runMultipleTests(t, myTests)
}
