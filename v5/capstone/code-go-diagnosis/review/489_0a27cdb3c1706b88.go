package templates

import "goacs/acs/types"

type Template struct {
	Id         int64               `db:"id" json:"id"`
	Name       string              `db:"name" json:"name"`
	Parameters []TemplateParameter `json:"parameters"`
}

type CPETemplate struct {
	Template
	Priority int64 `db:"priority" json:"priority"`
}

type TemplateParameter struct {
	UUID       string `db:"uuid" json:"uuid" `
	TemplateId int64  `db:"template_id" json:"template_id"`
	types.ParameterValueStruct
}

func (own *TemplateParameter) CompareTemplates(other []TemplateParameter) {
	//TODO: dorobić, pomyslec go-cmp

}
