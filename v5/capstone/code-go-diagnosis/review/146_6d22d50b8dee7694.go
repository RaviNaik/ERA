<gh_stars>0
package lib

import (
	"github.com/hailongz/kk-lib/dynamic"
	"github.com/hailongz/kk-logic/logic"
)

type AppLogic struct {
	logic.Logic
}

func init() {
	logic.Openlib("kk.Logic.App", func(object interface{}) logic.ILogic {
		v := AppLogic{}
		v.Init(object)
		return &v
	})
}

func (L *AppLogic) Exec(ctx logic.IContext, app logic.IApp) error {
	L.Logic.Exec(ctx, app)

	var a logic.IApp = nil
	var err error = nil

	path := dynamic.StringValue(L.Get(ctx, app, "path"), "")

	if path != "" {

		a, err = app.Store().Get(path)

		if err != nil {
			return err
		}

	} else {

		object := L.Get(ctx, app, "object")

		if object != nil {
			a = logic.NewApp(object, app.Store(), "<object>")
			defer a.Recycle()
		}
	}

	if a == nil {
		return L.Error(ctx, app, logic.NewError(logic.ERROR_UNKNOWN, "未找到应用"))
	}

	params := L.Get(ctx, app, logic.ParamsKeys[0])
	output := ctx.Get(logic.OutputKeys)
	ctx.Begin()
	ctx.Set(logic.OutletKeys, logic.Nil)
	ctx.Set(logic.ParamsKeys, params)
	ctx.Set(logic.ResultKeys, logic.Nil)
	ctx.Set(logic.OutputKeys, output)
	err = a.Exec(ctx, "in")
	v := ctx.Get(logic.ResultKeys)
	outlet := ctx.Get(logic.OutletKeys)

	if v == logic.Nil {
		v = nil
	}

	if outlet == logic.Nil {
		outlet = nil
	}

	ctx.End()

	if err != nil {
		if L.Has("error") {
			ctx.Set(logic.ErrorKeys, logic.GetErrorObject(err))
			return L.Done(ctx, app, "error")
		}
		return err
	}

	if v != nil {
		ctx.Set(logic.ResultKeys, v)
	}

	if outlet != nil {

		done := "done"

		a.Each(func(name string, v logic.ILogic) bool {

			if outlet == v {
				done = name
				return false
			}
			return true
		})

		return L.Done(ctx, app, "done")
	}

	return L.Done(ctx, app, "done")
}
