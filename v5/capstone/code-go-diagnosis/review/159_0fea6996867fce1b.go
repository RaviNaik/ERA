<filename>test/network_test.go
package test

import (
	"fmt"
	"strings"
	"testing"

	"github.com/gruntwork-io/terratest/modules/aws"
	"github.com/gruntwork-io/terratest/modules/random"
	"github.com/gruntwork-io/terratest/modules/terraform"
	test_structure "github.com/gruntwork-io/terratest/modules/test-structure"
	"github.com/stretchr/testify/require"
)

// initTestCases initializes a list of NetworkingModuleTestCase
func initTestCases() []NetworkingModuleTestCase {
	return []NetworkingModuleTestCase{
		{
			testName:         "Minimal",
			expectApplyError: false,
			vars: map[string]interface{}{
				"name_prefix":                   "minimal_terratest",
				"vpc_cidr_block":                "172.16.58.3/20",
				"ingress_cidr_blocks":           []string{"0.0.0.0/0"},
				"data_subnet_cidr_blocks":       []string{"172.16.58.3/24", "172.16.31.10/24"},
				"application_subnet_cidr_block": "192.168.3.11/24",
				"compute_subnet_cidr_block":     "172.16.58.3/24",
				"public_subnets_cidr_blocks":    []string{"172.16.31.10/24", "172.16.17.32/24"},
				"create_public_subnets":         false,
				"create_load_balancing_subnets": false,
				"enable_nat_gateway":            false,
				"tags":                          make(map[string]string),
			},
		},
		{
			testName:         "CreateAllSubnets",
			expectApplyError: false,
			vars: map[string]interface{}{
				"name_prefix":                        "all_subnets_terratest",
				"vpc_cidr_block":                     "172.16.58.3/20",
				"ingress_cidr_blocks":                []string{"0.0.0.0/0"},
				"data_subnet_cidr_blocks":            []string{"172.16.58.3/24", "172.16.31.10/24"},
				"application_subnet_cidr_block":      "192.168.3.11/24",
				"compute_subnet_cidr_block":          "172.16.58.3/24",
				"public_subnets_cidr_blocks":         []string{"172.16.31.10/24", "172.16.17.32/24"},
				"load_balancing_subnets_cidr_blocks": []string{"172.16.58.3/24", "172.16.31.10/24"},
				"create_public_subnets":              true,
				"create_load_balancing_subnets":      true,
				"enable_nat_gateway":                 true,
				"tags":                               make(map[string]string),
			},
		},
		{
			testName:         "InvalidCIDR",
			expectApplyError: true,
			vars: map[string]interface{}{
				"name_prefix":                   "this-should-fail",
				"vpc_cidr_block":                "0.0.0.0/0",
				"ingress_cidr_blocks":           []string{"0.0.0.0/0"},
				"data_subnet_cidr_blocks":       []string{"172.16.58.3/24", "172.16.31.10/24"},
				"application_subnet_cidr_block": "192.168.3.11/24",
				"compute_subnet_cidr_block":     "172.16.58.3/24",
				"create_public_subnets":         false,
				"create_load_balancing_subnets": false,
				"enable_nat_gateway":            false,
				"tags":                          make(map[string]string),
			},
		},
	}
}

// TestMinimalTamrNetwork runs all testCases
func TestTamrNetwork(t *testing.T) {
	// os.Setenv("TERRATEST_REGION", "us-east-1")

	// list of different buckets that will be created to be tested
	testCases := initTestCases()

	for _, testCase := range testCases {
		// The following is necessary to make sure testCase's values don't
		// get updated due to concurrency within the scope of t.Run(..) below
		testCase := testCase

		t.Run(testCase.testName, func(t *testing.T) {
			t.Parallel()
			awsRegion := aws.GetRandomStableRegion(t, []string{"us-east-1", "us-east-2", "us-west-1", "us-west-2"}, nil)
			// this creates a tempTestFolder for each testCase
			tempTestFolder := test_structure.CopyTerraformFolderToTemp(t, "..", "test_examples/minimal")

			expectedName := fmt.Sprintf("terratest-vpc-%s", strings.ToLower(random.UniqueId()))
			testCase.vars["tags"].(map[string]string)["Name"] = expectedName

			test_structure.RunTestStage(t, "setup_options", func() {
				terraformOptions := terraform.WithDefaultRetryableErrors(t, &terraform.Options{
					TerraformDir: tempTestFolder,
					Vars:         testCase.vars,
					EnvVars: map[string]string{
						"AWS_REGION": awsRegion,
					},
				})

				test_structure.SaveTerraformOptions(t, tempTestFolder, terraformOptions)
			})

			test_structure.RunTestStage(t, "create_network", func() {
				terraformOptions := test_structure.LoadTerraformOptions(t, tempTestFolder)

				_, err := terraform.InitAndApplyE(t, terraformOptions)

				if testCase.expectApplyError {
					require.Error(t, err)
					// If it failed as expected, we should skip the rest (validate function).
					t.SkipNow()
				}

				require.NoError(t, err)
			})

			defer test_structure.RunTestStage(t, "teardown", func() {
				teraformOptions := test_structure.LoadTerraformOptions(t, tempTestFolder)
				terraform.Destroy(t, teraformOptions)
			})

			test_structure.RunTestStage(t, "validate_network", func() {
				terraformOptions := test_structure.LoadTerraformOptions(t, tempTestFolder)
				validateNetwork(
					t,
					terraformOptions,
					awsRegion,
					expectedName,
					testCase.vars,
				)
			})

		})
	}
}
