package permission

import (
	"github.com/rwcoding/goback/models"
	"github.com/rwcoding/goback/pkg/api"
	"github.com/rwcoding/goback/pkg/boot"
)

type groupInfoRequest struct {
	ctx *boot.Context

	Id uint32 `validate:"required,numeric,min=1" json:"id"`
}

type groupInfoResponse struct {
	Id   uint32 `json:"id"`
	Name string `json:"name"`
	Ord  uint32 `json:"ord"`
}

func NewApiGroupInfo(ctx *boot.Context) boot.Logic {
	return &groupInfoRequest{ctx: ctx}
}

func (request *groupInfoRequest) Run() *api.Response {
	var p models.PermissionGroup
	if db().Take(&p, request.Id).Error != nil {
		return api.NewErrorResponse("无效的分组")
	}
	return api.NewDataResponse(&groupInfoResponse{
		Id:   p.Id,
		Name: p.Name,
		Ord:  p.Ord,
	})
}
