<reponame>hatingthefruit/rbldr<gh_stars>0
package util

import (
	"encoding/json"
	"io/ioutil"
	"os"
)

func NewResume(def *os.File, resumeDir string) Resume {
	resDecoder := json.NewDecoder(def)

	var resDef ResumeDefinition
	err := resDecoder.Decode(&resDef)
	CheckErr(err)
	resDef.Root = resumeDir

	var finalResume Resume

	contactFile, err := os.Open(resDef.Root + resDef.Contact + ".json")
	CheckErr(err)
	defer contactFile.Close()

	contactDecoder := json.NewDecoder(contactFile)
	contactDecoder.Decode(&finalResume.Contact)

	finalResume.Education = NewEducationList(resDef)
	finalResume.Experience = NewExperienceList(resDef)

	finalResume.Other = NewOtherMap(resDef)

	return finalResume
}

func NewEducationList(resDef ResumeDefinition) []Education {
	var eduList []Education
	for i := 0; i < len(resDef.Education); i++ {
		eduFile, err := os.Open(resDef.Root + resDef.Education[i] + ".json")
		defer eduFile.Close()
		CheckErr(err)

		var latestEdu Education
		json.NewDecoder(eduFile).Decode(&latestEdu)

		eduList = append(eduList, latestEdu)
	}
	return eduList
}

func NewExperienceList(resDef ResumeDefinition) []Experience {
	var expList []Experience
	for i := 0; i < len(resDef.Experience); i++ {
		expFile, err := os.Open(resDef.Root + resDef.Experience[i] + ".json")
		defer expFile.Close()
		CheckErr(err)

		var latestExp Experience
		json.NewDecoder(expFile).Decode(&latestExp)

		expList = append(expList, latestExp)
	}
	return expList
}

func NewOtherMap(resdef ResumeDefinition) map[string]map[string]string {
	otherMap := make(map[string]map[string]string)
	for _, section := range resdef.Other {
		rawJson, err := ioutil.ReadFile(resdef.Root + section + ".json")
		CheckErr(err)
		json.Unmarshal(rawJson, &otherMap)
	}
	return otherMap
}
