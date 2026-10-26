<reponame>open-devops/pipeline-daemon
package utility

import (
	"github.com/open-devops/pipeline-daemon/server/types"
	"os"
	"strings"
)

const (
	DEFAULT_ENGINE_PATH = "/opt/opendevops/pipelines"
	PIPELINE_ENGINE_NAM = "pipeline"
	PATH_SEPARATOR      = "/"
)

func GetEngineParentPath(pipelineInfo *types.PipelineInfo) string {
	return strings.Join(
		[]string{
			DEFAULT_ENGINE_PATH,
			pipelineInfo.ProductName,
			pipelineInfo.PipelineName,
		},
		PATH_SEPARATOR)
}

func GetEngineProgramPath(pipelineInfo *types.PipelineInfo) string {
	return strings.Join(
		[]string{
			DEFAULT_ENGINE_PATH,
			pipelineInfo.ProductName,
			pipelineInfo.PipelineName,
			PIPELINE_ENGINE_NAM,
		},
		PATH_SEPARATOR)
}

func GetContainerPrefix(name string) string {
	return strings.Replace(name, "-", "", -1)
}

func GetEngineTemplatePath() string {
	return os.Getenv("GOPATH") + "/src/github.com/open-devops/pipeline-docker/"
}
