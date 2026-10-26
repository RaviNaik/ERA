package gen

import (
	"io"
	"log"
	"os"
	"text/template"
)

func generate(wr io.Writer, templateFileName string, data map[string]string) error {
	t := template.Must(template.ParseFiles(templateFileName))
	err := t.Execute(wr, data)
	if err != nil {
		return err
	}

	return nil
}

// ParseTemplate 向指定writer写入渲染后的模板
func ParseTemplate(wr io.Writer, templateConfig TemplateConfig) error {
	for _, configElem := range templateConfig {
		err := generate(wr, configElem.TemplateFile, configElem.RenderData)
		if err != nil {
			log.Panic(err)
		}
	}
	return nil
}

// ParseTemplateWriteToFile 向配置的文件写入渲染后的模板
func ParseTemplateWriteToFile(templateConfig TemplateConfig) error {
	for _, configElem := range templateConfig {
		var f io.Writer
		var err error
		if configElem.OutputFile == "" {
			f = os.Stdout
		} else {
			f, err = os.Create(configElem.OutputFile)
			if err != nil {
				log.Panic(err)
			}
		}
		err = generate(f, configElem.TemplateFile, configElem.RenderData)
		if err != nil {
			log.Panic(err)
		}
	}
	return nil
}

func checkFileIsExist(filename string) bool {
	var exist = true
	if _, err := os.Stat(filename); os.IsNotExist(err) {
		exist = false
	}
	return exist
}
