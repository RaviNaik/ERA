package main

import (
	"log"
	"os"

	"github.com/davecgh/go-spew/spew"
	"github.com/huntresslabs/win-service-updater/updater"
)

func main() {
	info := updater.Info{}
	iuc, err := info.ParseWYC(os.Args[1])
	if nil != err {
		log.Fatal(err)
	}
	spew.Dump("%+v", iuc)
}
