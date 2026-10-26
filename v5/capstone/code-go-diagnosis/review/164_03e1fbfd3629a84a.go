<filename>entities/models/entitymodel.go
package models

import (
	"reflect"
	"strings"
)

// EntityModel - model of an entity in a database
type EntityModel struct {
	Table      string
	schematype SchemaType // type of schema
	entitytype reflect.Type
	columns    map[string]*ColumnDescriptor
	fields     map[string]*ColumnDescriptor
	indices    map[string]*IndexDescriptor
	uniques    map[string]*IndexDescriptor

	viewsql string // sql representing view if schema type is a view
}

// CreateViewModel creates an entity model with a view as source
//
// **Parameters**
//   - entitytype: type of entity reflected by view
//   - statement:  operation providing command text of view
//
// **Returns**
//   - *EntityModel: created entity model
func CreateViewModel(entitytype reflect.Type, statement string) *EntityModel {
	return &EntityModel{
		Table:      strings.ToLower(entitytype.Name()),
		schematype: SchemaTypeView,
		viewsql:    statement}
}

// CreateModel - creates a new entity model for a type
func CreateModel(entitytype reflect.Type) *EntityModel {
	var model EntityModel
	model.Table = strings.ToLower(entitytype.Name())
	model.schematype = SchemaTypeTable
	model.entitytype = entitytype
	model.columns = make(map[string]*ColumnDescriptor)
	model.fields = make(map[string]*ColumnDescriptor)
	model.indices = make(map[string]*IndexDescriptor)
	model.uniques = make(map[string]*IndexDescriptor)

	var indices map[string][]string = make(map[string][]string)
	var uniques map[string][]string = make(map[string][]string)

	for i := 0; i < entitytype.NumField(); i++ {
		field := entitytype.Field(i)

		descriptor := ColumnDescriptor{
			name:     strings.ToLower(field.Name),
			field:    field.Name,
			datatype: field.Type}

		var tag string = field.Tag.Get("database")
		if len(tag) > 0 {
			var options []string = strings.Split(tag, ",")
			for _, option := range options {
				switch option {
				case "primarykey":
					descriptor.isprimarykey = true
				case "autoincrement":
					descriptor.isautoincrement = true
				case "unique":
					descriptor.isunique = true
				case "notnull":
					descriptor.isnotnull = true
				default:
					if strings.HasPrefix(option, "column=") {
						descriptor.name = option[7:]
					} else if strings.HasPrefix(option, "index=") {
						var indexname = option[6:]

						if _, exists := indices[indexname]; !exists {
							indices[indexname] = make([]string, 0)
						}

						indices[indexname] = append(indices[indexname], descriptor.name)
					} else if strings.HasPrefix(option, "unique=") {
						var uniquename = option[7:]

						if _, exists := uniques[uniquename]; !exists {
							uniques[uniquename] = make([]string, 0)
						}
						uniques[uniquename] = append(uniques[uniquename], descriptor.name)
					} else if strings.HasPrefix(option, "default=") {
						descriptor.defaultvalue = option[8:]
					}
				}
			}
		}

		model.columns[descriptor.name] = &descriptor
		model.fields[field.Name] = &descriptor
	}

	for key, value := range indices {
		model.indices[key] = NewIndexDescriptor(key, value...)
	}

	for key, value := range uniques {
		model.uniques[key] = NewIndexDescriptor(key, value...)
	}

	return &model
}

// CreateModelWithTable - creates a new entity model for a type
func CreateModelWithTable(entitytype reflect.Type, table string) *EntityModel {
	model := CreateModel(entitytype)
	model.Table = table
	return model
}

// Columns - access to all columns in model
func (model *EntityModel) Columns() []*ColumnDescriptor {
	var columns []*ColumnDescriptor

	for _, value := range model.columns {
		columns = append(columns, value)
	}

	return columns
}

// Indices index definitions of entity model
//
// **Returns**
//   - []*IndexDescriptor: index definitions
func (model *EntityModel) Indices() []*IndexDescriptor {
	indices := make([]*IndexDescriptor, 0)
	for _, value := range model.indices {
		indices = append(indices, value)
	}
	return indices
}

// Uniques index definitions of entity model
//
// **Returns**
//   - []*IndexDescriptor: index definitions
func (model *EntityModel) Uniques() []*IndexDescriptor {
	uniques := make([]*IndexDescriptor, 0)
	for _, value := range model.uniques {
		uniques = append(uniques, value)
	}
	return uniques
}

// Column - provides access to a column descriptor by column name
func (model *EntityModel) Column(columnname string) *ColumnDescriptor {
	return model.columns[columnname]
}

// ColumnFromField - provides access to a column descriptor by field name
func (model *EntityModel) ColumnFromField(fieldname string) *ColumnDescriptor {
	return model.fields[fieldname]
}

// EntityType - get entity type information used to create model
func (model *EntityModel) EntityType() reflect.Type {
	return model.entitytype
}

// ViewSQL sql used to create view
//
// **Returns**
//   - string: view sql string
func (model *EntityModel) ViewSQL() string {
	return model.viewsql
}
