<filename>command/translatableerror/service_plan_not_found_error.go
package translatableerror

type ServicePlanNotFoundError struct {
	PlanName     string
	OfferingName string
}

func (e ServicePlanNotFoundError) Error() string {
	if e.OfferingName == "" {
		return "Service plan '{{.PlanName}}' not found."
	}
	return "The plan {{.PlanName}} could not be found for service {{.OfferingName}}."
}

func (e ServicePlanNotFoundError) Translate(translate func(string, ...interface{}) string) string {
	return translate(e.Error(), map[string]interface{}{
		"PlanName":     e.PlanName,
		"OfferingName": e.OfferingName,
	})
}
