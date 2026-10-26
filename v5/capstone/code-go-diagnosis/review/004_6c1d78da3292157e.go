package main

import (
	"fmt"
	"gin_demo/log"
	gormadapter "github.com/casbin/gorm-adapter/v2"
	"github.com/gin-gonic/gin"
)

func SetupRouter() *gin.Engine {
	r := gin.Default()
	r.GET("/ping", func(c *gin.Context) {
		c.String(200, "pong")
	})
	return r
}

func init() {
	Adapter, err := gormadapter.NewAdapter("mysql", "root:@tcp(127.0.0.1:3306)/casbin")
	if err != nil {
		panic(err)
	}
	fmt.Printf("Adapter:%v\n", Adapter)
}

func main() {
	r := SetupRouter()
	r.Use(log.LogFormatter())
	r.Run(":8080")
}