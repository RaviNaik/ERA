package controllers

import (
	"encoding/json"
	"time"

	. "github.com/gtechx/base/common"
	"github.com/gtechx/chatserver/db"
)

type MyAppDataController struct {
	BaseController
}

func (c *MyAppDataController) Prepare() {
	c.BaseController.Prepare()
	c.Data["isreadonly"] = true
	c.Data["nav"] = "usermyappdata"
}

func (c *MyAppDataController) Index() {
	applist, _ := gtdb.Manager().GetAccountAppList(c.account)
	c.Data["applist"] = applist
	c.TplName = "appdata.tpl"
}

func (c *MyAppDataController) List() {
	index := Int(c.GetString("pageNumber")) - 1 //Int(c.Ctx.Input.Param("0"))
	pagesize := Int(c.GetString("pageSize"))    //Int(c.Ctx.Input.Param("1"))

	id, _ := c.GetUint64("id", 0)
	appname := c.GetString("appname")
	zonename := c.GetString("zonename")
	account := c.GetString("account")
	if account != c.account {
		account = c.account
	}

	appdatafilter := &gtdb.AppDataFilter{}
	appdatafilter.Nickname = c.GetString("nickname")
	appdatafilter.Desc = c.GetString("desc")
	appdatafilter.Sex = c.GetString("sex")
	appdatafilter.Country = c.GetString("country")
	appdatafilter.Regip = c.GetString("regip")
	appdatafilter.Lastip = c.GetString("lastip")
	// lastloginbegindate := c.GetString("lastloginbegindate")
	// lastloginenddate := c.GetString("lastloginenddate")
	// createbegindate := c.GetString("createbegindate")
	// createenddate := c.GetString("createenddate")

	bbdate, err := time.Parse("01/02/2006", c.GetString("birthdaybegindate"))
	if err == nil {
		appdatafilter.Birthdaybegindate = &bbdate
	}
	bedate, err := time.Parse("01/02/2006", c.GetString("birthdayenddate"))
	if err == nil {
		appdatafilter.Birthdayenddate = &bedate
	}
	lbdate, err := time.Parse("01/02/2006", c.GetString("lastloginbegindate"))
	if err == nil {
		appdatafilter.Lastloginbegindate = &lbdate
	}
	ledate, err := time.Parse("01/02/2006", c.GetString("lastloginenddate"))
	if err == nil {
		appdatafilter.Lastloginenddate = &ledate
	}
	cbdate, err := time.Parse("01/02/2006", c.GetString("createbegindate"))
	if err == nil {
		appdatafilter.Createbegindate = &cbdate
	}
	cedate, err := time.Parse("01/02/2006", c.GetString("createenddate"))
	if err == nil {
		appdatafilter.Createenddate = &cedate
	}

	println("pageNumber:", index, " pageSize:", pagesize)

	dataManager := gtdb.Manager()

	if id != 0 {
		appdata, err := dataManager.GetAppData(id)

		if err != nil {
			println(err.Error())
			c.Ctx.Output.Body([]byte("[]"))
			return
		}

		pageapp := PageData{Total: 1, Rows: []*gtdb.AppData{appdata}}
		retjson, err := json.Marshal(pageapp)
		if err != nil {
			println(err.Error())
			c.Ctx.Output.Body([]byte("[]"))
			return
		}

		c.Ctx.Output.Body(retjson)
		return
	}

	if appname == "" {
		println("appname must not null")
		c.Ctx.Output.Body([]byte("[]"))
		return
	}

	totalcount, err := dataManager.GetMyAppDataCount(appname, zonename, account, appdatafilter)

	if err != nil {
		println(err.Error())
		c.Ctx.Output.Body([]byte("[]"))
		return
	}

	if totalcount == 0 {
		c.Ctx.Output.Body([]byte("[]"))
		return
	}

	appdatalist, err := dataManager.GetMyAppDataList(appname, zonename, account, index*pagesize, index*pagesize+pagesize-1, appdatafilter)

	if err != nil {
		println(err.Error())
		c.Ctx.Output.Body([]byte("[]"))
		return
	}

	pageapp := PageData{Total: totalcount, Rows: appdatalist}
	retjson, err := json.Marshal(pageapp)
	if err != nil {
		println(err.Error())
		c.Ctx.Output.Body([]byte("[]"))
		return
	}

	c.Ctx.Output.Body(retjson)
}

func (c *MyAppDataController) ZoneList() {
	appname := c.GetString("appname")
	account := c.GetString("account")
	if account != c.account {
		account = c.account
	}
	zonelist, err := gtdb.Manager().GetAccountZoneList(account, appname)

	if err != nil {
		println(err.Error())
		c.Ctx.Output.Body([]byte("[]"))
		return
	}

	pagezone := PageData{Total: uint64(len(zonelist)), Rows: zonelist}
	retjson, err := json.Marshal(pagezone)
	if err != nil {
		println(err.Error())
		c.Ctx.Output.Body([]byte("[]"))
		return
	}

	c.Ctx.Output.Body(retjson)
}
