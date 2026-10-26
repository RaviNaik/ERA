// Copyright © 2019 NAME HERE <EMAIL ADDRESS>
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package cmd

import (
	"fmt"
	"bufio"
	"os"
	"github.com/spf13/cobra"
	"github.com/spf13/viper"
	"strings"
	"os/exec"
	"io"
	"github.com/kr/pty"
	"gopkg.in/yaml.v2"
)

var cfgFile string

var rootCmd = &cobra.Command{
	Use:   "daenv",
	DisableFlagParsing: true,
	Short: "Sources the environment for Tasker",
	Run: func(command *cobra.Command, args []string) {
		sysCmd := getSysCmd(args)
		env := getEnv(args)
		envFile := getEnvFile(env)
		cmdArgs := getCmdArgs(args)
		mergeConfigs(envFile)
		cmd := exec.Command(sysCmd, cmdArgs...)
		cmd.Env = envVars()
	 	cmd.Stdout = os.Stdout
  	cmd.Stderr = os.Stderr
    cmd.Stdin = os.Stdin
		f, err := pty.Start(cmd)
		if err != nil {
			panic(err)
		}
		defer f.Close()
		io.Copy(os.Stdout, f)
		go func() {
			scanner := bufio.NewScanner(f)
			for scanner.Scan() {
				fmt.Println("[" + sysCmd + "] " + scanner.Text())
			}
    }()
    go func() {
      io.Copy(f, os.Stdin)
    }()
		cmd.Run()
		cmd.Wait()
	},
}

func Execute() {
	if err := rootCmd.Execute(); err != nil {
		fmt.Println(err)
		os.Exit(1)
	}
}

func init() {
	cobra.OnInitialize(initConfig)
}

// initConfig reads in config file and ENV variables if set.
func initConfig() {
	if cfgFile != "" {
		// Use config file from the flag.
		viper.SetConfigFile(cfgFile)
	} else {
		// Find home directory.
		dir, err := os.Getwd()
		if err != nil {
			fmt.Println(err)
			os.Exit(1)
		}
		// Search config in home directory with name ".daenv" (without extension).
		viper.AddConfigPath(dir)
		viper.SetConfigName(".daenv")
	}

	readViper()
}

func mergeConfigs(env_file string) {
	viper.SetConfigName(env_file)
	readViper()
}

func readViper() {
	if err := viper.ReadInConfig(); err != nil {
		fmt.Println(err)
	}
}

func envVars() []string {
	c := viper.AllSettings()
	bs, err := yaml.Marshal(c)
	if err != nil {
		fmt.Println(err)
	}

	vs := strings.Split(string(bs), "\n")
	env := os.Environ()
	for _ , v := range vs {
		split_value := strings.Split(v, ": ")
		if len(split_value) == 2 {
			key := strings.ToUpper(split_value[0])
			value := strings.Replace(split_value[1], "\"", "", -1)
			env = append(env, key + "=" + value)
		}
	}
	return env
}

func getSysCmd(args []string) string {
	return args[0]
}

func getCmdArgs(args []string) []string {
	b := []string{}
	for i, x := range args {
		if i != 0 && !strings.Contains(x, "--env") {
			b = append(b, x)
		}
	}

	return b
}

func getEnv(args []string) string {
	index := -1
	for i, v := range args {
		if strings.Contains(v, "--env") {
			index = i
		}
  }
	env := "dev"
	if index > 0 {
		envString := strings.Split(args[index], "=")
		if len(env) > 1 {
			env = envString[1]
		}
	}
	return env
}

func getEnvFile(env string) string {
	envFile := viper.GetString("DEV_ENV")
	switch env {
	case "dev":
		envFile = viper.GetString("DEV_ENV")
	case "prod":
		envFile = viper.GetString("PROD_ENV")
	case "test":
		envFile = viper.GetString("TEST_ENV")
	}

	return envFile
}
