<filename>user/main.go<gh_stars>0
package main

import (
	"log"

	"github.com/defsky/bookstore/micro"
	userPb "github.com/defsky/bookstore/proto/user"
	"github.com/defsky/bookstore/user/handler"
)

func main() {
	s := micro.NewService(
		micro.Name("com.afkplayer.service.user"),
		micro.Version("latest"),
	).Init()

	userPb.RegisterUserServer(s.S, handler.UserHandler)

	// Run service
	if err := s.Serve(); err != nil {
		log.Fatalln(err)
	}
}
